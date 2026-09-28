"""Isolated native ComfyUI H3 cross-host experiment; no production edits.

Uses buqi-code Ulysses attention/block primitives, preserving installed H3
embedding, final heads, audio scaling, sampler, and VAEs. Experimental.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
from datetime import timedelta
from types import MethodType

ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['link', 'parity', 'benchmark', 'generate'], required=True)
ap.add_argument('--rank', type=int, default=0)
ap.add_argument('--world', type=int, choices=[1,2], default=1)
ap.add_argument('--master', default='127.0.0.1')
ap.add_argument('--port', type=int, default=29628)
ap.add_argument('--comfy-root', required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--model', default='minimax_h3_ref2va_int8_convrot.safetensors')
ap.add_argument('--width', type=int, default=832)
ap.add_argument('--height', type=int, default=480)
ap.add_argument('--frames', type=int, default=56)
ap.add_argument('--steps', type=int, default=20)
ap.add_argument('--repeats', type=int, default=1)
ap.add_argument('--seed', type=int, default=20260928)
ap.add_argument('--exchange', choices=['alltoall', 'allgather'], default='alltoall')
ap.add_argument('--attention-chunks', type=int, default=1)
ap.add_argument('--gather-chunks', type=int, default=4)
ap.add_argument('--benchmark-shapes', default='320x192x22,832x480x56,1280x736x56')
ap.add_argument('--generate-after-benchmark', action='store_true')
ap.add_argument('--parallel-vae', action='store_true')
ap.add_argument('--prompt', default='A cinematic medium shot of a small red sailboat floating on a calm lake at sunrise. The boat gently rocks, soft ripples spread across the water, reeds sway in a light breeze. The camera slowly moves forward. Natural warm light, realistic water reflections, continuous motion. Audio: gentle water lapping and a soft breeze, no music, no speech.')
a = ap.parse_args()
if not 0 <= a.rank < a.world:
    ap.error('rank must be in [0, world)')
if a.mode in ('link', 'parity', 'benchmark') and a.world != 2:
    ap.error('link/parity/benchmark require --world 2')
if a.world == 2 and a.master == '127.0.0.1':
    ap.error('set --master to the head node fabric address for --world 2')
if any(v <= 0 for v in (a.width, a.height, a.frames, a.steps, a.repeats,
                        a.attention_chunks, a.gather_chunks)):
    ap.error('dimensions, steps, repeats and chunk counts must be positive')
if a.width % 16 or a.height % 16:
    ap.error('width and height must be multiples of 16')
try:
    comfy_commit = subprocess.check_output(
        ['git', '-C', a.comfy_root, 'rev-parse', 'HEAD'], text=True).strip()
except (OSError, subprocess.CalledProcessError):
    comfy_commit = 'unknown'
outdir = Path(a.output)
outdir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
sys.path.insert(0, a.comfy_root)
# Explicit Comfy flags; parsed only by the private process.
sys.argv = [sys.argv[0], '--use-ck-attention', '--highvram', '--disable-pinned-memory', '--preview-method', 'none']
import comfy.options
comfy.options.enable_args_parsing()
import torch
import torch.distributed as dist
torch.cuda.set_device(0)
torch.set_num_threads(8)
device = torch.device('cuda', 0)
obj_pg = None

def record(name, data):
    p = outdir / name
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps({'event': name, 'data': data}, ensure_ascii=False), flush=True)

def init_dist():
    global obj_pg
    dist.init_process_group('nccl', init_method=f'tcp://{a.master}:{a.port}',
                            rank=a.rank, world_size=a.world,
                            timeout=timedelta(minutes=8), device_id=device)
    obj_pg = dist.new_group(backend='gloo', timeout=timedelta(minutes=8))

def sync():
    torch.cuda.synchronize()

def link_test():
    results = []
    for mb in [1, 16, 64]:
        t = torch.ones(mb * 1024 * 1024 // 4, device=device)
        dist.all_reduce(t)
        assert bool((t == 2).all())
        dist.barrier(); sync()
        start = time.perf_counter()
        for _ in range(12):
            dist.all_reduce(t)
        sync()
        seconds = (time.perf_counter() - start) / 12
        results.append({'MB': mb, 'seconds': seconds, 'algorithm_GBps': t.numel()*4/seconds/1e9})
    record(f'link_rank{a.rank}.json', {'host': socket.gethostname(), 'torch': torch.__version__,
                                     'nccl': torch.cuda.nccl.version(), 'results': results})

def install_blocks(dit):
    import sp_primitives as spf
    spf.ATTN_CHUNKS = a.attention_chunks
    spf.AG_CHUNKS = a.gather_chunks
    state = {}
    originals = []
    for index, block in enumerate(dit.blocks):
        originals.append(block.forward)
        def distributed_block(self, x, t_emb, mod_segments, rope_freqs,
                              transformer_options=None, attention=None, _index=index):
            options = transformer_options or {}
            if attention is not None or options.get('patches_replace'):
                raise RuntimeError('This experimental adapter does not support custom block replacements')
            if _index == 0:
                state['ctx'] = spf.SPContext(a.rank, a.world, None, x.shape[0])
                ctx = state['ctx']
                x = x[ctx.start:ctx.stop].contiguous()
            ctx = state['ctx']
            if any(isinstance(row, torch.Tensor) for _, _, row in mod_segments):
                raise RuntimeError('Per-row denoise masks are not yet supported')
            segments = spf.shard_segments(mod_segments, ctx.start, ctx.stop)
            rope = rope_freqs if a.exchange == 'allgather' else rope_freqs[:, ctx.start:ctx.stop].contiguous()
            x = spf.sp_block(self, x, t_emb, segments, rope, ctx, options,
                             allgather=a.exchange == 'allgather')
            if _index == len(dit.blocks)-1:
                x = spf.gather_rows_full(x, ctx)
            return x
        block.forward = MethodType(distributed_block, block)
    return originals

def load_dit():
    import comfy.model_management as mm
    import comfy.sd
    import folder_paths
    t = time.perf_counter()
    path = folder_paths.get_full_path_or_raise('diffusion_models', a.model)
    patcher = comfy.sd.load_diffusion_model(path)
    mm.load_models_gpu([patcher], force_full_load=True)
    sync()
    record(f'model_rank{a.rank}.json', {'path': path, 'bytes': os.stat(path).st_size,
           'load_seconds': time.perf_counter()-t, 'allocated_GiB': torch.cuda.memory_allocated()/2**30})
    return patcher

def parity(patcher):
    dit = patcher.model.diffusion_model
    from comfy_extras.nodes_minimax_h3 import temporal_shape
    _, lat_t, audio_t = temporal_shape(a.frames)
    torch.manual_seed(a.seed)
    x = [torch.randn(1,24,lat_t,a.height//16,a.width//16,device=device),
         torch.randn(1,32,2,audio_t,device=device)]
    context = torch.randn(1,64,5120,device=device,dtype=torch.bfloat16)
    timestep = torch.tensor([500.0],device=device)
    reference = None
    with torch.no_grad():
        if a.rank == 0:
            sync(); t=time.perf_counter()
            reference = dit._forward(x,timestep,context,transformer_options={})
            sync(); native_s=time.perf_counter()-t
        dist.barrier()
        install_blocks(dit)
        sync(); t=time.perf_counter()
        got = dit._forward(x,timestep,context,transformer_options={})
        sync(); parallel_s=time.perf_counter()-t
        if a.rank == 0:
            errors=[]
            for ref,g in zip(reference,got):
                delta=(ref.float()-g.float()).abs()
                errors.append({'max_abs':delta.max().item(), 'relative_max':delta.max().item()/max(ref.abs().max().item(),1e-12),
                               'rms':delta.square().mean().sqrt().item(), 'finite':bool(torch.isfinite(g).all())})
            record('parity.json', {'native_seconds':native_s,'parallel_seconds':parallel_s,'errors':errors,
                                  'note':'Synthetic one-forward diagnostic, not an end-to-end speed benchmark'})
            assert all(e['finite'] and e['relative_max'] < 0.02 for e in errors), 'Parity gate failed'
    dist.barrier()

def benchmark(patcher):
    """Warm same-weight forward A/B; excludes encoder, decoder and cold load."""
    from comfy_extras.nodes_minimax_h3 import temporal_shape
    dit = patcher.model.diffusion_model
    all_results = []
    with torch.no_grad():
        for shape in a.benchmark_shapes.split(','):
            width, height, frames = map(int, shape.split('x'))
            _, lat_t, audio_t = temporal_shape(frames)
            torch.manual_seed(a.seed)
            x = [torch.randn(1,24,lat_t,height//16,width//16,device=device),
                 torch.randn(1,32,2,audio_t,device=device)]
            context = torch.randn(1,128,5120,device=device,dtype=torch.bfloat16)
            timestep = torch.tensor([500.0],device=device)
            entry = {'shape':shape, 'native_s':[], 'variants':[]}
            reference = None
            if a.rank == 0:
                for repeat in range(4):
                    sync(); start=time.perf_counter()
                    reference=dit._forward(x,timestep,context,transformer_options={})
                    sync(); elapsed=time.perf_counter()-start
                    if repeat: entry['native_s'].append(elapsed)
                    print(f'BENCH {shape} native {repeat} {elapsed:.3f}s',flush=True)
            dist.barrier()
            originals=install_blocks(dit)
            for exchange, chunks, gathers in [('alltoall',1,4),('allgather',1,1),('allgather',1,2),('allgather',1,4),('allgather',4,1),('allgather',4,4)]:
                a.exchange, a.attention_chunks, a.gather_chunks = exchange, chunks, gathers
                import sp_primitives as spf
                spf.ATTN_CHUNKS=chunks
                spf.AG_CHUNKS=gathers
                variant={'exchange':exchange,'attention_chunks':chunks,'gather_chunks':gathers,'seconds':[]}
                for repeat in range(4):
                    dist.barrier(); sync(); start=time.perf_counter()
                    got=dit._forward(x,timestep,context,transformer_options={})
                    sync(); elapsed=time.perf_counter()-start
                    if repeat: variant['seconds'].append(elapsed)
                    if a.rank==0:
                        print(f'BENCH {shape} {exchange}/heads{chunks}/gathers{gathers} {repeat} {elapsed:.3f}s',flush=True)
                if a.rank==0:
                    errors=[]
                    for ref,g in zip(reference,got):
                        ref,g=ref.float(),g.float()
                        delta=ref-g
                        errors.append({'relative_max':float(delta.abs().max()/ref.abs().max()),
                                       'relative_l2':float(delta.norm()/ref.norm()),
                                       'reference_rms':float(ref.square().mean().sqrt()),
                                       'finite':bool(torch.isfinite(g).all())})
                    variant['errors']=errors
                    variant['numerical_gate']=all(e['finite'] and e['relative_max']<0.02 and e['relative_l2']<0.01 for e in errors)
                    entry['variants'].append(variant)
                    record('benchmark_partial.json',all_results+[entry])
            for block,original in zip(dit.blocks,originals): block.forward=original
            all_results.append(entry)
            if a.rank==0: record('benchmark.json',all_results)
            dist.barrier()
    if a.generate_after_benchmark:
        selected=[None]
        if a.rank==0:
            valid=[v for v in all_results[-1]['variants'] if v['numerical_gate']]
            if valid:
                best=min(valid,key=lambda v:statistics.median(v['seconds']))
                selected[0]=(best['exchange'],best['attention_chunks'],best['gather_chunks'])
        dist.broadcast_object_list(selected,src=0,group=obj_pg)
        if selected[0] is None: raise RuntimeError('No parallel variant passed numerical gate')
        a.exchange,a.attention_chunks,a.gather_chunks=selected[0]
        if a.rank: worker(patcher)
        else: generate(patcher)

def worker(patcher):
    import sp_protocol as spg
    dit=patcher.model.diffusion_model
    install_blocks(dit)
    count=0
    video_vae=None
    while True:
        box=[None]; dist.broadcast_object_list(box,src=0,group=obj_pg)
        meta=box[0]
        if meta['op']=='shutdown': break
        if meta['op']=='vae_decode':
            import nodes
            import parallel_vae
            if video_vae is None:
                video_vae=nodes.VAELoader().load_vae('minimax_h3_video_vae_fp16.safetensors')[0]
            with torch.no_grad():
                chunks=parallel_vae.worker_decode(video_vae,meta,device)
            print(f'RANK1_VAE_CHUNKS_DONE {chunks}',flush=True)
            continue
        video=spg.alloc_from_meta(meta['video'],device)
        audio=spg.alloc_from_meta(meta['audio'],device)
        timestep=spg.alloc_from_meta(meta['timestep'],device)
        context=spg.alloc_from_meta(meta['context'],device)
        ts={'tags':spg.alloc_from_meta(meta['tags'],device),
            'cond_video':[spg.alloc_from_meta(m,device) for m in meta['cond_video']],
            'cond_audio':[spg.alloc_from_meta(m,device) for m in meta['cond_audio']]}
        for t in [video,audio,timestep,context,ts['tags'],*ts['cond_video'],*ts['cond_audio']]:
            if t is not None: dist.broadcast(t,src=0)
        opts=meta['options']
        if 'sample_sigmas_list' in meta:
            opts['sample_sigmas']=torch.tensor(meta['sample_sigmas_list'],device=device)
        with torch.no_grad():
            dit._forward([video,audio],timestep,context,transformer_options=opts,
                         minimax_payload=spg.payload_from_meta(meta,ts))
        count+=1
        print(f'RANK1_FORWARD_DONE {count}',flush=True)
    record('worker_complete.json', {'host':socket.gethostname(),'forwards':count})

def attach_dispatch(patcher):
    import sp_protocol as spg
    dit=patcher.model.diffusion_model
    install_blocks(dit)
    native=dit._forward
    def dispatch(x,timestep,context,transformer_options=None,minimax_payload=None,**kwargs):
        options=transformer_options or {}
        for name in ('denoise_mask','audio_denoise_mask'):
            mask=kwargs.get(name)
            if mask is not None and not bool((mask==1).all()):
                raise RuntimeError('Nontrivial masks not yet qualified for distributed adapter')
        meta=spg.build_meta(x,timestep,context,options,minimax_payload)
        if options.get('sample_sigmas') is not None:
            meta['sample_sigmas_list']=options['sample_sigmas'].detach().cpu().tolist()
        dist.broadcast_object_list([meta],src=0,group=obj_pg)
        for t in spg.ordered_tensors(x,timestep,context,minimax_payload):
            if t is not None: dist.broadcast(t.to(device).contiguous(),src=0)
        return native(x,timestep,context,transformer_options=options,minimax_payload=minimax_payload,**kwargs)
    patcher.add_object_patch('diffusion_model._forward',dispatch)

def generate(patcher):
    import nodes
    import comfy.model_management as mm
    import comfy.samplers
    import folder_paths
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, RandomNoise, SamplerCustomAdvanced
    from comfy_extras.nodes_audio import VAEDecodeAudio
    from comfy_extras.nodes_video import CreateVideo
    if a.world>1: attach_dispatch(patcher)
    clip=nodes.CLIPLoader().load_clip('qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors','minimax','default')[0]
    vv=nodes.VAELoader().load_vae('minimax_h3_video_vae_fp16.safetensors')[0]
    va=nodes.VAELoader().load_vae('minimax_h3_audio_vae_fp32.safetensors')[0]
    record('config.json',vars(a)|{'host':socket.gethostname(),'torch':torch.__version__,
                               'comfy_commit':comfy_commit,
                               'lora':None,'sampler':'res_multistep','scheduler':'simple','attention':'comfy-kitchen'})
    results=[]
    with torch.no_grad():
        for run in range(a.repeats):
            sync(); total_start=time.perf_counter(); start=total_start
            cond,latent=MiniMaxH3ReferenceToVideo.execute(clip,a.prompt,a.width,a.height,a.frames).result
            sync(); encode_s=time.perf_counter()-start
            guider=BasicGuider.execute(patcher,cond).result[0]
            sigmas=BasicScheduler.execute(patcher,'simple',a.steps,1.0).result[0]
            noise=RandomNoise.execute(a.seed).result[0]
            sync(); start=time.perf_counter()
            samples=SamplerCustomAdvanced.execute(noise,guider,comfy.samplers.sampler_object('res_multistep'),sigmas,latent).result[0]
            sync(); sample_s=time.perf_counter()-start
            start=time.perf_counter()
            if a.parallel_vae and a.world>1:
                import parallel_vae
                z=samples['samples'].unbind()[0]
                images=parallel_vae.distributed_decode(vv,z,obj_pg,device)
                if len(images.shape)==5: images=images.reshape(-1,*images.shape[-3:])
            else:
                images=nodes.VAEDecode().decode(vv,samples)[0]
            audio=VAEDecodeAudio.execute(va,samples).result[0]
            sync(); decode_s=time.perf_counter()-start
            start=time.perf_counter()
            video=CreateVideo.execute(images,24.0,audio,bit_depth=8).result[0]
            path=outdir/f'h3_world{a.world}_run{run:02d}.mp4'
            video.save_to(str(path),format='mp4',codec='h264')
            sync()
            item={'run':run,'conditioning_s':encode_s,'sampling_s':sample_s,'decode_s':decode_s,
                  'save_s':time.perf_counter()-start,'total_s':time.perf_counter()-total_start,
                  'file':str(path),'frames':images.shape[0], 'width':images.shape[2], 'height':images.shape[1],
                  'gpu_peak_GiB':torch.cuda.max_memory_allocated()/2**30}
            results.append(item); record('results.json',results)
            if a.parallel_vae and run==0:
                # Verify outside the recorded generation timer: same final
                # sampled latent through the untouched native decoder.
                reference_images=nodes.VAEDecode().decode(vv,samples)[0]
                max_abs=float((images-reference_images).abs().max())
                record('vae_parity.json',{'exact':bool(torch.equal(images,reference_images)),
                                          'max_abs':max_abs,'shape':list(images.shape)})
                if max_abs>1e-5: raise RuntimeError('Parallel VAE pixel parity gate failed')
                del reference_images
            del cond,latent,guider,samples,images,audio,video
            gc.collect()
    if a.world>1: dist.broadcast_object_list([{'op':'shutdown'}],src=0,group=obj_pg)

try:
    if a.world>1: init_dist()
    if a.mode=='link': link_test()
    else:
        import sp_primitives, sp_protocol  # Fail fast before multi-minute weight load.
        patcher=load_dit()
        if a.world>1: dist.barrier()
        if a.mode=='parity': parity(patcher)
        elif a.mode=='benchmark': benchmark(patcher)
        elif a.rank: worker(patcher)
        else: generate(patcher)
    if a.world>1: dist.destroy_process_group()
    record(f'exit_rank{a.rank}.json',{'exit':0,'host':socket.gethostname()})
except BaseException as e:
    record(f'failure_rank{a.rank}.json',{'error':repr(e),'host':socket.gethostname()})
    raise
