"""Isolated reproduction of the current T01 Singularity two-pass compute graph.

Keeps model, LoRAs, AdaLN port, sparse/dense policy, FF chunks, sigma split,
latent upscaler and native VAEs. Saves with native CreateVideo to the run folder
instead of the UI-only VHS output node. No production service changes.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import logging
import statistics
from pathlib import Path
import sys
import time
from datetime import timedelta
from types import MethodType

p = argparse.ArgumentParser()
p.add_argument('--rank', type=int, default=0)
p.add_argument('--world', type=int, choices=[1, 2], default=1)
p.add_argument('--master', default='127.0.0.1')
p.add_argument('--port', type=int, default=29629)
p.add_argument('--comfy-root', required=True)
p.add_argument('--output', required=True)
p.add_argument('--width', type=int, default=768)
p.add_argument('--height', type=int, default=448)
p.add_argument('--frames', type=int, default=56)
p.add_argument('--seed', type=int, default=20260928)
p.add_argument('--repeats', type=int, default=2)
p.add_argument('--reference-video')
p.add_argument('--reference-tensors', help='Trusted local tensors exported by the ComfyUI node')
p.add_argument('--prompt', help='Override the built-in no-reference or reference sailboat prompt')
p.add_argument('--compare-dir')
p.add_argument('--compare-run-zero', action='store_true', help='Same fixed seed each repeat: compare every repeat to baseline run00')
p.add_argument('--exchange', choices=['alltoall', 'allgather'], default='alltoall')
p.add_argument('--attention-chunks', type=int, default=1)
p.add_argument('--gather-chunks', type=int, default=4)
p.add_argument('--parallel-vae', action='store_true')
p.add_argument('--keep-stage-qkv', action='store_true', help='Retain separate immutable effective QKV copies for Turbo and LMS')
p.add_argument('--tune', action='store_true', help='Measure exact-output chunk candidates once per stage; tuning run is not a benchmark')
p.add_argument('--tune-stage', choices=['all', 'turbo', 'lms'], default='all')
p.add_argument('--tune-grid', choices=['standard', 'wide'], default='standard')
p.add_argument('--tune-repeats', type=int, default=2)
p.add_argument('--turbo-chunks', nargs=2, type=int, metavar=('GATHER', 'ATTENTION'))
p.add_argument('--lms-chunks', nargs=2, type=int, metavar=('GATHER', 'ATTENTION'))
a = p.parse_args()
if not 0 <= a.rank < a.world:
    p.error('rank must be in [0, world)')
if not 1024 <= a.port <= 65535:
    p.error('port must be between 1024 and 65535')
if min(a.width, a.height, a.frames, a.repeats) < 1 or a.width % 16 or a.height % 16:
    p.error('positive dimensions/repeats required; width and height must be multiples of 16')
if a.parallel_vae and a.world != 2:
    p.error('--parallel-vae requires world 2')
if a.compare_run_zero and not a.compare_dir:
    p.error('--compare-run-zero requires --compare-dir')
if a.tune and (a.world != 2 or a.exchange != 'allgather'):
    p.error('--tune requires world 2 and allgather')
if any(n < 1 for n in [a.attention_chunks, a.gather_chunks, *(a.turbo_chunks or []), *(a.lms_chunks or [])]):
    p.error('Chunk counts must be positive')
if a.tune_repeats < 1:
    p.error('Tuning requires at least one timed repeat')
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
sys.path.insert(0, a.comfy_root)
sys.argv = [sys.argv[0], '--use-ck-attention', '--reserve-vram', '12', '--disable-pinned-memory', '--preview-method', 'none']
import comfy.options
comfy.options.enable_args_parsing()
from comfy.cli_args import args as comfy_args
import comfy_aimdo.control
# Native ComfyUI main.py performs these steps before model construction.
# Importing nodes alone leaves the legacy patcher active, regardless of flags.
try:
    comfy_aimdo.control.init(simple_vram_headroom=int(comfy_args.reserve_vram * 1024**3),
                             nvml_pressure=not comfy_args.disable_nvml_pressure)
except TypeError:
    try:
        comfy_aimdo.control.init(simple_vram_headroom=int(comfy_args.reserve_vram * 1024**3))
    except TypeError:
        comfy_aimdo.control.init()
import torch
import torch.distributed as dist
import comfy.model_management as mm
import comfy.model_patcher
import comfy.memory_management
try:
    dynamic_ok = comfy_aimdo.control.init_devices(
        (d.index, int(comfy_args.vram_headroom * 1024**3)) for d in mm.get_all_torch_devices())
except TypeError:
    dynamic_ok = comfy_aimdo.control.init_devices(d.index for d in mm.get_all_torch_devices())
if not dynamic_ok:
    raise RuntimeError('Native dynamic VRAM initialization failed')
comfy_aimdo.control.set_log_info()
comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
comfy.memory_management.aimdo_enabled = True
logging.info('Native dynamic VRAM explicitly initialized')
import nodes
import sp_primitives as spf
import sp_protocol as protocol
import stage_qkv
stage_qkv.retain_stages = a.keep_stage_qkv
torch.cuda.set_device(0)
torch.set_num_threads(8)
device = torch.device('cuda', 0)
obj_pg = None
models = {}
original_forward = None
forward_counts = {'turbo': 0, 'lms': 0}
tuned_settings = {}
tuning_results = {}


def record(name, data):
    (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({'event': name, 'data': data}, ensure_ascii=False, default=str), flush=True)


def sync():
    torch.cuda.synchronize()


def set_chunks(settings):
    spf.AG_CHUNKS, spf.ATTN_CHUNKS = settings


def stage_settings(stage):
    return tuned_settings.get(stage) or getattr(a, stage + '_chunks') or [a.gather_chunks, a.attention_chunks]


def run_forward(stage, x, timestep, context, opts, payload, kwargs, settings):
    set_chunks(settings)
    meta = protocol.build_meta(x, timestep, context, opts, payload)
    meta.update(stage=stage, chunks=list(settings))
    for key in ('sample_sigmas', 'sigmas'):
        if isinstance(opts.get(key), torch.Tensor):
            meta[key + '_list'] = opts[key].detach().cpu().tolist()
    dist.broadcast_object_list([meta], src=0, group=obj_pg)
    for tensor in protocol.ordered_tensors(x, timestep, context, payload):
        if tensor is not None:
            dist.broadcast(tensor.to(device).contiguous(), src=0)
    forward_counts[stage] += 1
    return original_forward(x, timestep, context, transformer_options=copy.copy(opts),
        minimax_payload=payload, **kwargs)


def tune_stage(stage, x, timestep, context, opts, payload, kwargs):
    # Reuse the actual sampler input, effective LoRA and backend. Each candidate
    # has one untimed warm-up and synchronized measurements. Tuning overhead
    # remains in this run's timings, and is never reported as generation speed.
    reference = run_forward(stage, x, timestep, context, opts, payload, kwargs, [4, 4])
    reference = [v.detach().clone() for v in reference]
    rows = []
    candidates = ([4, 4], [1, 1], [2, 1], [4, 1], [1, 4], [2, 4])
    if a.tune_grid == 'wide':
        candidates = ([4, 4], [8, 4], [16, 4], [4, 7], [8, 7], [8, 14], [4, 4])
    for settings in candidates:
        elapsed, exact = [], True
        for repeat in range(a.tune_repeats + 1):
            sync(); start = time.perf_counter()
            got = run_forward(stage, x, timestep, context, opts, payload, kwargs, settings)
            sync(); seconds = time.perf_counter() - start
            equal = len(got) == len(reference) and all(torch.equal(v, ref) for v, ref in zip(got, reference))
            exact = exact and equal
            if repeat:
                elapsed.append(seconds)
            del got
        rows.append({'gather_chunks': settings[0], 'attention_chunks': settings[1],
                     'seconds': elapsed, 'median_s': statistics.median(elapsed), 'exact': exact})
        record('tuning_' + stage + '.json', {'candidates': rows})
    qualified = [r for r in rows if r['exact']]
    if not qualified:
        raise RuntimeError('No exact-output communication candidate for ' + stage)
    best = min(qualified, key=lambda row: row['median_s'])
    tuned_settings[stage] = [best['gather_chunks'], best['attention_chunks']]
    tuning_results[stage] = {'candidates': rows, 'selected': tuned_settings[stage]}
    record('tuning_results.json', tuning_results)


def load_module(name, relative, package=False):
    path = Path(a.comfy_root) / relative
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_models():
    global original_forward
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend
    from comfy_extras.nodes_sparse_attention import BlockSparseAttention
    kj = load_module('lab_kj_minimax', 'custom_nodes/ComfyUI-KJNodes/nodes/minimax_nodes.py')
    adaln = load_module('lab_adaln', 'custom_nodes/ComfyUI-H3-AdaLN-LoRA-Fix/__init__.py', True)
    # The node's basis fit performs CPU matrix operations. A loader may
    # place the live embedder on CUDA, unlike the UI load order.
    # Materialize only the four small basis inputs on CPU; retain native math.
    adaln_math = adaln.adaln_node.adaln_patch
    math_module = sys.modules['lab_adaln.adaln']
    math_module._write_cache = lambda *args, **kwargs: None  # production models/ stays read-only
    def grid_on_cpu(embedder):
        tensors = [embedder.proj_in.weight, embedder.proj_in.bias,
                   embedder.proj_out.weight, embedder.proj_out.bias]
        record(f'embedder_rank{a.rank}.json', [{'device': str(t.device), 'dtype': str(t.dtype),
            'type': type(t).__name__, 'shape': list(t.shape)} for t in tensors])
        return math_module.silu_temb_grid(*[t.detach().float().cpu() for t in tensors],
            freq_dim=int(getattr(embedder, 'freq_dim', 256))).clone()
    adaln_math.grid_from_time_embedder = grid_on_cpu
    start = time.perf_counter()
    model = nodes.UNETLoader().load_unet('Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors', 'default')[0]
    model = ModelAttentionBackend.execute(model, 'comfy kitchen attention').result[0]
    model = BlockSparseAttention.execute(model, {'selection': 'sol-attn', 'tau': 1.3}, 0.2, 0.9,
        dense_blocks='', min_tokens=1048576, extra_tokens=256, sink_conditioning='exact_kv_and_rows', verbose=True).result[0]
    model = kj.MiniMaxChunkFeedForward.execute(model, 2, 4096).result[0]
    turbo = nodes.LoraLoaderModelOnly().load_lora_model_only(model,
        'minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors', 1.0)[0]
    lms = nodes.LoraLoaderModelOnly().load_lora_model_only(turbo, 'minimax_h3_lms_v1.0_r64.safetensors', 0.3)[0]
    lms, report = adaln.adaln_node.adaln_patch.fix_model(lms, 'port')
    record(f'adaln_rank{a.rank}.json', report)
    if report.get('ported') != 50 or report.get('effective_mode') != 'port' or report.get('unportable') or report.get('stripped'):
        raise RuntimeError('LMS AdaLN did not port all 50 groups')
    models.update(turbo=turbo, lms=lms)
    original_forward = turbo.model.diffusion_model._forward
    if a.keep_stage_qkv:
        stage_qkv.install_prefetch_filter(turbo.model.diffusion_model.blocks)
    if a.world == 2:
        attach_parallel_blocks()
        if a.rank == 0:
            for stage, patcher in models.items():
                def dispatch(x, timestep, context, transformer_options=None, minimax_payload=None, _stage=stage, **kwargs):
                    stage_qkv.activate(_stage)
                    opts = transformer_options or {}
                    for name in ('denoise_mask', 'audio_denoise_mask'):
                        mask = kwargs.get(name)
                        if mask is not None and not bool((mask == 1).all()):
                            raise RuntimeError('Nontrivial mask is outside this validation')
                    if a.tune and _stage not in tuned_settings and a.tune_stage in ('all', _stage):
                        tune_stage(_stage, x, timestep, context, opts, minimax_payload, kwargs)
                    return run_forward(_stage, x, timestep, context, opts, minimax_payload, kwargs,
                                       stage_settings(_stage))
                patcher.add_object_patch('diffusion_model._forward', dispatch)
    elif a.keep_stage_qkv:
        # Apply the same reusable effective-weight cache to the single-rank
        # control; all native attention and block math remain unchanged.
        for stage, patcher in models.items():
            def cached_forward(x, timestep, context, transformer_options=None, _stage=stage, **kwargs):
                stage_qkv.activate(_stage)
                forward_counts[_stage] += 1
                return original_forward(x, timestep, context, transformer_options=transformer_options, **kwargs)
            patcher.add_object_patch('diffusion_model._forward', cached_forward)
            for index, block in enumerate(patcher.model.diffusion_model.blocks):
                def cached_projection(proj, x, _stage=stage):
                    stage_qkv.activate(_stage)
                    return stage_qkv.effective_head_shard(proj, 1, 0, x)(x)
                patcher.add_object_patch(f'diffusion_model.blocks.{index}.attn.qkv_proj.forward',
                                         MethodType(cached_projection, block.attn.qkv_proj))
    record(f'models_rank{a.rank}.json', {'load_and_build_s': time.perf_counter() - start,
        'patcher_type': type(turbo).__name__, 'dynamic_vram': turbo.is_dynamic(),
        'patch_counts': {k: len(v.patches) for k, v in models.items()}, 'world': a.world})


def attach_parallel_blocks():
    dit = models['turbo'].model.diffusion_model
    state = {}
    spf.ATTN_CHUNKS = a.attention_chunks
    spf.AG_CHUNKS = a.gather_chunks
    for index, block in enumerate(dit.blocks):
        def parallel(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None, attention=None, _index=index):
            opts = transformer_options or {}
            # The native sparse wrapper has already decided whether to inject
            # an attention callback. Preserve its dense decision; fail if active.
            if attention is not None:
                raise RuntimeError('Active sparse attention needs a separate distributed implementation')
            if _index == 0:
                if x.shape[0] >= 1048576:
                    raise RuntimeError('This experiment only qualifies the template dense fallback')
                state['ctx'] = spf.SPContext(a.rank, a.world, None, x.shape[0])
                ctx = state['ctx']
                print(f'RANK{a.rank}_SEQUENCE full={x.shape[0]} owned={ctx.stop-ctx.start}', flush=True)
                x = x[ctx.start:ctx.stop].contiguous()
            ctx = state['ctx']
            if any(isinstance(row, torch.Tensor) for _, _, row in mod_segments):
                raise RuntimeError('Per-row masks are not qualified')
            segments = spf.shard_segments(mod_segments, ctx.start, ctx.stop)
            rope = rope_freqs if a.exchange == 'allgather' else rope_freqs[:, ctx.start:ctx.stop].contiguous()
            result = spf.sp_block(self, x, t_emb, segments, rope,
                                  ctx, opts, allgather=a.exchange == 'allgather')
            return spf.gather_rows_full(result, ctx) if _index == len(dit.blocks)-1 else result
        bound = MethodType(parallel, block)
        for patcher in models.values():
            patcher.add_object_patch(f'diffusion_model.blocks.{index}.forward', bound)


def worker():
    current = None
    video_vae = None
    vae_chunks = 0
    while True:
        box = [None]
        dist.broadcast_object_list(box, src=0, group=obj_pg)
        meta = box[0]
        if meta['op'] == 'shutdown':
            break
        if meta['op'] == 'vae_decode':
            import parallel_vae
            if video_vae is None:
                video_vae = nodes.VAELoader().load_vae('minimax_h3_video_vae_int8_convrot.safetensors')[0]
            stage_qkv.activate(None)
            vae_chunks += parallel_vae.worker_decode(video_vae, meta, device)
            current = None
            print(f'RANK1_VAE_CHUNKS_DONE total={vae_chunks}', flush=True)
            continue
        stage = meta['stage']
        set_chunks(meta['chunks'])
        if stage != current:
            start = time.perf_counter()
            mm.load_models_gpu([models[stage]])
            sync()
            current = stage
            stage_qkv.activate(stage)
            print(f'RANK1_STAGE {stage} loaded_s={time.perf_counter()-start:.3f}', flush=True)
        video = protocol.alloc_from_meta(meta['video'], device)
        audio = protocol.alloc_from_meta(meta['audio'], device)
        timestep = protocol.alloc_from_meta(meta['timestep'], device)
        context = protocol.alloc_from_meta(meta['context'], device)
        ts = {'tags': protocol.alloc_from_meta(meta['tags'], device),
              'cond_video': [protocol.alloc_from_meta(m, device) for m in meta['cond_video']],
              'cond_audio': [protocol.alloc_from_meta(m, device) for m in meta['cond_audio']]}
        for tensor in [video, audio, timestep, context, ts['tags'], *ts['cond_video'], *ts['cond_audio']]:
            if tensor is not None:
                dist.broadcast(tensor, src=0)
        opts = copy.copy(models[stage].model_options['transformer_options'])
        opts.update(meta['options'])
        for key in ('sample_sigmas', 'sigmas'):
            if key + '_list' in meta:
                opts[key] = torch.tensor(meta[key + '_list'], device=device)
        original_forward([video, audio], timestep, context, transformer_options=opts,
                         minimax_payload=protocol.payload_from_meta(meta, ts))
        forward_counts[stage] += 1
        print('RANK1_FORWARD_DONE', stage, forward_counts[stage], flush=True)
    record('worker_complete.json', forward_counts | {'vae_chunks': vae_chunks})


def reference_file(name, suffix):
    if a.compare_run_zero:
        name = name.rsplit('_run', 1)[0] + '_run00'
    return Path(a.compare_dir) / (name + suffix)


def tensor_evidence(samples, name):
    tensors = [t.detach().cpu() for t in samples['samples'].unbind()]
    torch.save(tensors, out / (name + '.pt'))
    evidence = {'shapes': [list(t.shape) for t in tensors], 'finite': [bool(torch.isfinite(t).all()) for t in tensors]}
    if a.compare_dir:
        path = reference_file(name, '.pt')
        reference = torch.load(path, map_location='cpu', weights_only=True)
        if len(reference) != len(tensors) or any(ref.shape != got.shape for ref, got in zip(reference, tensors)):
            raise RuntimeError('Reference latent shape mismatch')
        evidence['comparison_reference'] = str(path)
        evidence['comparison'] = []
        for ref, got in zip(reference, tensors):
            delta = ref.float() - got.float()
            evidence['comparison'].append({'exact': torch.equal(ref, got), 'max_abs': float(delta.abs().max()),
                'relative_l2': float(delta.norm() / ref.float().norm().clamp_min(1e-12))})
    record(name + '_latents.json', evidence)
    if not all(evidence['finite']):
        raise RuntimeError('Nonfinite latent')
    if a.compare_dir and not all(row['exact'] for row in evidence['comparison']):
        raise RuntimeError('Reference latent differs')


def save_video(images, audio, name):
    from comfy_extras.nodes_video import CreateVideo
    pixels = images.detach().cpu().contiguous()
    evidence = {'shape': list(pixels.shape), 'finite': bool(torch.isfinite(pixels).all()),
        'sha256_float_pixels': hashlib.sha256(pixels.numpy().tobytes()).hexdigest()}
    if a.compare_dir:
        reference = json.loads(reference_file(name, '_pixels.json').read_text())
        evidence['reference_pixels_exact'] = evidence['sha256_float_pixels'] == reference['sha256_float_pixels']
        if not evidence['reference_pixels_exact']:
            record(name + '_pixels.json', evidence)
            raise RuntimeError('Final decoded float pixels differ from native baseline')
    record(name + '_pixels.json', evidence)
    video = CreateVideo.execute(images, 24.0, audio, bit_depth=8).result[0]
    video.save_to(str(out / (name + '.mp4')), format='mp4', codec='h264')


def decode_video(vae, samples):
    if a.parallel_vae and a.world == 2:
        import parallel_vae
        stage_qkv.activate(None)
        video = samples['samples'].unbind()[0]
        images = parallel_vae.distributed_decode(vae, video, obj_pg, device)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        return images
    return nodes.VAEDecode().decode(vae, samples)[0]


def generate():
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, RandomNoise, DisableNoise, SamplerCustomAdvanced, ExtendIntermediateSigmas, SplitSigmas
    from comfy_extras.nodes_lt import LTXVSeparateAVLatent, LTXVConcatAVLatent
    from comfy_extras.nodes_audio import VAEDecodeAudio
    import comfy.samplers
    up = load_module('lab_h3_upscale', 'custom_nodes/Comfyui_Minimax_h3_latent_Upscaler/nodes/minimax_h3_latent_upscaler_3d.py')
    clip = nodes.CLIPLoader().load_clip('qwen3vl_32b_minimax_h3_int8_convrot.safetensors', 'minimax', 'default')[0]
    vv = nodes.VAELoader().load_vae('minimax_h3_video_vae_int8_convrot.safetensors')[0]
    va = nodes.VAELoader().load_vae('minimax_h3_audio_vae_fp32.safetensors')[0]
    prompt = 'A cinematic medium shot of a small red sailboat floating on a calm lake at sunrise. The boat gently rocks, soft ripples spread across the water, reeds sway in a light breeze. The camera slowly moves forward. Natural warm light, realistic water reflections, continuous motion. Audio: gentle water lapping and a soft breeze, no music, no speech.'
    refs = {}
    if a.reference_video:
        import av
        import numpy as np
        with av.open(a.reference_video) as container:
            frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
        tensor = torch.from_numpy(np.stack(frames)).float() / 255
        refs = {'ref_images': {'ref_image_0': tensor[:1]}, 'ref_videos': {'ref_video_0': tensor}}
        prompt = 'subject_definitions:\n<Subject 1> is the red sailboat in <Picture 1>. <Picture 1> defines its appearance. <Video 1> provides gentle boat and camera motion.\nscene_description:\nA calm lake at sunrise, warm natural light and reeds at the edge.\ncamera_and_motion:\nOne continuous medium shot with a slow forward camera movement. The red sailboat gently rocks and ripples move across the water.\ndialogue_and_audio:\nGentle water lapping and a soft breeze. No speech, no music.\nconstraints:\nOne boat only. No cuts, no text, no extra people.'
    if a.reference_tensors:
        refs = torch.load(a.reference_tensors, map_location='cpu', weights_only=True)
    if a.prompt is not None:
        prompt = a.prompt
    record('config.json', vars(a) | {'prompt': prompt, 'sampler': 'euler', 'scheduler': 'simple', 'base_steps': 6,
        'extend_steps': 2, 'split_steps': [2, 0], 'turbo': 1.0, 'lms': 0.3, 'adaln': 'port', 'scale': 1.5,
        'align': 32, 'ref_image_size': 'max', 'backend': 'comfy-kitchen', 'ff_chunks': 2,
        'sparse_min_tokens': 1048576, 'parallel_exchange': a.exchange, 'video_vae_parallel': a.parallel_vae})
    results = []
    sampler = comfy.samplers.sampler_object('euler')
    for run in range(a.repeats):
        timings = {'run': run}
        sync(); total = time.perf_counter(); start = total
        cond, latent = MiniMaxH3ReferenceToVideo.execute(clip, prompt, a.width, a.height, a.frames,
            ref_image_size='max', vae=vv, audio_vae=va, **refs).result
        sync(); timings['conditioning_s'] = time.perf_counter() - start
        low = BasicGuider.execute(models['turbo'], cond).result[0]
        high = BasicGuider.execute(models['lms'], cond).result[0]
        sigmas = BasicScheduler.execute(models['turbo'], 'simple', 6, 1.0).result[0]
        sigmas = ExtendIntermediateSigmas.execute(sigmas, 2, 1.0000000000000002, 0, 'linear').result[0]
        first_sigmas, rest_sigmas = SplitSigmas.execute(sigmas, 2).result
        zero_sigmas = SplitSigmas.execute(rest_sigmas, 0).result[0]
        record(f'sigmas_run{run:02d}.json', {'all': sigmas.tolist(), 'first': first_sigmas.tolist(), 'zero': zero_sigmas.tolist(), 'second': rest_sigmas.tolist()})
        noise = RandomNoise.execute(a.seed).result[0]
        no_noise = DisableNoise.execute().result[0]
        sync(); start = time.perf_counter()
        first = SamplerCustomAdvanced.execute(noise, low, sampler, first_sigmas, latent).result
        sync(); timings['first_sampling_s'] = time.perf_counter() - start
        start = time.perf_counter()
        first_images = decode_video(vv, first[1])
        first_audio = VAEDecodeAudio.execute(va, first[0]).result[0]
        sync(); timings['first_decode_s'] = time.perf_counter() - start
        start = time.perf_counter()
        save_video(first_images, first_audio, f'first_run{run:02d}')
        timings['first_save_s'] = time.perf_counter() - start
        del first_images, first_audio
        tensor_evidence(first[1], f'first_run{run:02d}')
        start = time.perf_counter()
        video_latent = LTXVSeparateAVLatent.execute(first[1]).result[0]
        audio_latent = LTXVSeparateAVLatent.execute(first[0]).result[1]
        enlarged = up.MinimaxH3LatentUpscaler3D.execute(video_latent,
            'minimax_h3_latent_upscaler_3d_bf16.safetensors', {'mode': 'scale by multiplier', 'scale': 1.5},
            32, False, True, 'cuda', 'bf16').result[0]
        initialized = SamplerCustomAdvanced.execute(noise, low, sampler, zero_sigmas, enlarged).result[0]
        second_input = LTXVConcatAVLatent.execute(initialized, audio_latent).result[0]
        sync(); timings['upscale_and_zero_step_s'] = time.perf_counter() - start
        start = time.perf_counter()
        second = SamplerCustomAdvanced.execute(no_noise, high, sampler, rest_sigmas, second_input).result[1]
        sync(); timings['second_sampling_s'] = time.perf_counter() - start
        start = time.perf_counter()
        images = decode_video(vv, second)
        audio = VAEDecodeAudio.execute(va, second).result[0]
        sync(); timings['second_decode_s'] = time.perf_counter() - start
        start = time.perf_counter()
        save_video(images, audio, f'second_run{run:02d}')
        timings['second_save_s'] = time.perf_counter() - start
        sync(); timings['total_s_including_evidence'] = time.perf_counter() - total
        timings['output_shape'] = list(images.shape)
        results.append(timings)
        record('results.json', results)
        tensor_evidence(second, f'second_run{run:02d}')
        del cond, latent, low, high, first, video_latent, audio_latent, enlarged, initialized, second_input, second, images, audio
        gc.collect()
    if a.world == 2:
        dist.broadcast_object_list([{'op': 'shutdown'}], src=0, group=obj_pg)


def main():
    global obj_pg
    try:
        if a.world == 2:
            dist.init_process_group('nccl', init_method=f'tcp://{a.master}:{a.port}', rank=a.rank, world_size=2,
                                    timeout=timedelta(minutes=12), device_id=device)
            obj_pg = dist.new_group(backend='gloo', timeout=timedelta(minutes=12))
        with torch.no_grad():
            build_models()
            if a.world == 2:
                dist.barrier()
            if a.rank:
                worker()
            else:
                generate()
        record(f'exit_rank{a.rank}.json', {'exit': 0, 'forward_counts': forward_counts, 'qkv_cache': stage_qkv.stats})
        if a.world == 2:
            dist.destroy_process_group()
    except BaseException as error:
        record(f'failure_rank{a.rank}.json', {'error': repr(error)})
        raise


if __name__ == "__main__":
    main()
