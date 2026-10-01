"""Execute the actual author graph, retaining native loaders, samplers and outputs.

The existing qualified parallel kernels live in the separate, pinned legacy backend.
All method wrappers below exist only in this owned child process.
"""
import argparse
import asyncio
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from datetime import timedelta

p = argparse.ArgumentParser()
for key in ('comfy-root', 'legacy-backend', 'output', 'graph-file', 'master'):
    p.add_argument('--' + key, required=True)
p.add_argument('--world', type=int, choices=(1, 2), required=True)
p.add_argument('--rank', type=int, required=True)
p.add_argument('--port', type=int, required=True)
args = p.parse_args()
out = Path(args.output)
spec = json.loads(Path(args.graph_file).read_text())
plan = spec['plan']
sys.path.insert(0, args.legacy_backend)
sys.argv = ['singularity_runner.py', '--comfy-root', args.comfy_root, '--output', args.output,
            '--master', args.master, '--port', str(args.port), '--world', str(args.world),
            '--rank', str(args.rank), '--repeats', '1', '--exchange', 'allgather',
            '--attention-chunks', '4', '--gather-chunks', '4']
if args.world == 2:
    sys.argv += ['--parallel-vae', '--keep-stage-qkv']
import singularity_runner as r
import torch
import torch.distributed as dist
import nodes
import folder_paths

if args.world == 2 and os.environ.get('H3_S02_FIXED_CUBLAS_LMS') == '1':
    import fixed_cublas
    fixed_cublas.install(r.spf, r)


def evidence(name, tensors):
    if not spec.get('evidence') and not spec.get('validation_dir'):
        return
    values = [t.detach().contiguous().cpu() for t in tensors]
    row = {'shapes': [list(t.shape) for t in values],
           'finite': [bool(torch.isfinite(t).all()) for t in values],
           'sha256': [hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest() for t in values]}
    if spec.get('validation_dir'):
        ref = json.loads((Path(spec['validation_dir']) / (name + '.json')).read_text())
        row['exact'] = row['sha256'] == ref['sha256'] and row['shapes'] == ref['shapes']
        if not row['exact']:
            r.record(name + '.json', row)
            raise RuntimeError('作者图同输入单机对照不一致: ' + name)
    r.record(name + '.json', row)


def read_only_adaln():
    # Same numerical port as the deployed node; keep its caches out of models/.
    module = r.load_module('author_adaln', 'custom_nodes/ComfyUI-H3-AdaLN-LoRA-Fix/__init__.py', True)
    math = sys.modules['author_adaln.adaln']
    math._write_cache = lambda *a, **k: None
    def grid_cpu(embedder):
        tensors = [embedder.proj_in.weight, embedder.proj_in.bias, embedder.proj_out.weight, embedder.proj_out.bias]
        return math.silu_temb_grid(*[t.detach().float().cpu() for t in tensors],
                                  freq_dim=int(getattr(embedder, 'freq_dim', 256))).clone()
    module.adaln_node.adaln_patch.grid_from_time_embedder = grid_cpu
    node_class = module.adaln_node.H3AdaLNLoRAFix
    nodes.NODE_CLASS_MAPPINGS[node_class.GET_SCHEMA().node_id] = node_class


async def initialize():
    from server import PromptServer
    from app.assets.manager import default_asset_manager
    server = PromptServer(asyncio.get_running_loop(), default_asset_manager())
    server.client_id = 'author-child'
    events = (out / 'events.jsonl').open('w')
    def send(event, data, sid=None):
        if isinstance(event, str):
            events.write(json.dumps({'event': event, 'data': data, 'time': time.time()},
                                   ensure_ascii=False, default=str) + '\n')
            events.flush()
    server.send_sync = send
    # main.start_comfyui normally installs this hook; this child intentionally
    # starts no HTTP server, so wire actual sampler/VAE steps explicitly.
    from comfy_execution.progress import get_progress_state
    from comfy_execution.utils import get_executing_context
    from progress_relay import make_progress_hook
    import comfy.utils
    import comfy.model_management
    comfy.utils.set_progress_bar_global_hook(make_progress_hook(
        server, get_executing_context, get_progress_state,
        comfy.model_management.throw_exception_if_processing_interrupted))
    # Use the same existing custom node implementations; do not start a web server.
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    for directory in ('ComfyUI-KJNodes', 'rgthree-comfy', 'Comfyui_Minimax_h3_latent_Upscaler',
                      'ComfyUI-VideoHelperSuite', 'Goohaitools-comfyui'):
        if not await nodes.load_custom_node(str(Path(args.comfy_root) / 'custom_nodes' / directory)):
            raise RuntimeError('Required installed custom node failed: ' + directory)
    read_only_adaln()
    return server, events


def executor(server):
    import execution
    if args.world == 2:
        # PromptExecutor creates weights inside inference_mode, whose tensors
        # lack the version counters required for safe retained-QKV invalidation.
        # Only this isolated executor uses versioned no-grad tensors; never
        # replace torch.inference_mode globally or weaken cache signatures.
        class VersionedTorch:
            def __getattr__(self, name):
                return getattr(torch, name)

            @staticmethod
            @contextmanager
            def inference_mode():
                with torch.inference_mode(False), torch.no_grad():
                    yield

        execution.torch = VersionedTorch()
    return execution.PromptExecutor(server, cache_type=execution.CacheType.CLASSIC,
                                    cache_args={'ram': 16, 'ram_inactive': 16, 'lru': 0})


async def execute_graph(server, graph, outputs):
    import execution
    valid, error, selected, node_errors = await execution.validate_prompt(spec['job'], graph, outputs)
    if not valid or set(outputs) - set(selected):
        r.record('validation_error.json', {'error': error, 'nodes': node_errors})
        raise RuntimeError('原作者图验证失败，见 validation_error.json')
    e = executor(server)
    await e.execute_async(graph, spec['job'], {'client_id': 'author-child',
                                             'extra_pnginfo': spec.get('extra_pnginfo', {})}, outputs)
    if not e.success or not any(kind == 'execution_success' for kind, _ in e.status_messages):
        r.record('execution_error.json', e.status_messages)
        raise RuntimeError('原作者图执行失败，见 execution_error.json')
    return e


async def worker_models(server):
    graph = copy.deepcopy(plan['model_graph'])
    original_stack = nodes.NODE_CLASS_MAPPINGS['Lora Loader Stack (rgthree)']
    class WorkerStack:
        @classmethod
        def INPUT_TYPES(cls):
            inputs = copy.deepcopy(original_stack.INPUT_TYPES())
            for group in inputs.values():
                group.pop('clip', None)
            return inputs
        RETURN_TYPES = ('MODEL', 'CLIP')
        FUNCTION = 'load'
        def load(self, model, **kw):
            return original_stack().load_lora(model, None, **kw)
    nodes.NODE_CLASS_MAPPINGS['Lora Loader Stack (rgthree)'] = WorkerStack
    class Capture:
        @classmethod
        def INPUT_TYPES(cls):
            return {'required': {'model': ('MODEL',), 'stage': ('STRING',)}}
        RETURN_TYPES = ()
        OUTPUT_NODE = True
        FUNCTION = 'capture'
        def capture(self, model, stage):
            r.models[stage] = model
            return ()
    nodes.NODE_CLASS_MAPPINGS['_AuthorCapture'] = Capture
    outputs = []
    for stage, model_id in plan['model_outputs'].items():
        key = '_capture_' + stage
        outputs.append(key)
        graph[key] = {'class_type': '_AuthorCapture', 'inputs': {'model': [model_id, 0], 'stage': stage}}
    await execute_graph(server, graph, outputs)
    r.original_forward = r.models['turbo'].model.diffusion_model._forward
    r.stage_qkv.install_prefetch_filter(r.models['turbo'].model.diffusion_model.blocks)
    r.attach_parallel_blocks()
    r.record('worker_model_plan.json', plan)


def install_head_wrappers():
    from comfy_extras.nodes_custom_sampler import BasicGuider, SamplerCustomAdvanced
    original_vae = nodes.VAEDecode
    original_audio = nodes.NODE_CLASS_MAPPINGS['VAEDecodeAudio']
    class AuthorGuider:
        @classmethod
        def INPUT_TYPES(cls):
            return {'required': {'model': ('MODEL',), 'conditioning': ('CONDITIONING',)},
                    'hidden': {'unique_id': 'UNIQUE_ID'}}
        RETURN_TYPES = ('GUIDER',)
        FUNCTION = 'make'
        def make(self, model, conditioning, unique_id):
            if args.world == 1:
                return BasicGuider.execute(model, conditioning).result
            stage = plan['guider_stages'][unique_id]
            patcher = model.clone()
            if r.original_forward is None:
                r.original_forward = model.model.diffusion_model._forward
                r.stage_qkv.install_prefetch_filter(model.model.diffusion_model.blocks)
            r.models[stage] = patcher
            r.models.setdefault('turbo', patcher)
            r.attach_parallel_blocks()
            def dispatch(x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs):
                for name in ('denoise_mask', 'audio_denoise_mask'):
                    mask = kwargs.get(name)
                    if mask is not None and not bool((mask == 1).all()):
                        raise RuntimeError('当前双机尚未验证非均匀采样遮罩')
                r.stage_qkv.activate(stage)
                return r.run_forward(stage, x, timestep, context, transformer_options or {}, minimax_payload, kwargs, [4, 4])
            patcher.add_object_patch('diffusion_model._forward', dispatch)
            return BasicGuider.execute(patcher, conditioning).result
    class AuthorSampler:
        @classmethod
        def INPUT_TYPES(cls):
            return {'required': {'noise': ('NOISE',), 'guider': ('GUIDER',), 'sampler': ('SAMPLER',),
                                 'sigmas': ('SIGMAS',), 'latent_image': ('LATENT',)},
                    'hidden': {'unique_id': 'UNIQUE_ID'}}
        RETURN_TYPES = ('LATENT', 'LATENT')
        FUNCTION = 'sample'
        def sample(self, unique_id, **kw):
            start = time.monotonic()
            result = SamplerCustomAdvanced.execute(**kw).result
            for index, latent in enumerate(result):
                samples = latent['samples']
                values = list(samples.unbind()) if samples.is_nested else [samples]
                evidence('sampler_' + unique_id + '_' + str(index), values)
            r.record('timing_sampler_' + unique_id + '.json', {'seconds': time.monotonic() - start,
                                                             'sigmas': kw['sigmas'].cpu().tolist()})
            return result
    class AuthorVAE:
        @classmethod
        def INPUT_TYPES(cls):
            return {'required': {'samples': ('LATENT',), 'vae': ('VAE',)},
                    'hidden': {'unique_id': 'UNIQUE_ID'}}
        RETURN_TYPES = ('IMAGE',)
        FUNCTION = 'decode'
        def decode(self, vae, samples, unique_id):
            start = time.monotonic()
            if args.world == 2:
                import parallel_vae
                r.stage_qkv.activate(None)
                video = samples['samples']
                if video.is_nested:
                    video = video.unbind()[0]
                images = parallel_vae.distributed_decode(vae, video, r.obj_pg, r.device)
                if images.ndim == 5:
                    images = images.reshape(-1, *images.shape[-3:])
            else:
                images = original_vae().decode(vae, samples)[0]
            evidence('pixels_' + unique_id, [images])
            r.record('timing_vae_' + unique_id + '.json', {'seconds': time.monotonic() - start})
            return (images,)
    nodes.NODE_CLASS_MAPPINGS.update(BasicGuider=AuthorGuider, SamplerCustomAdvanced=AuthorSampler, VAEDecode=AuthorVAE)


async def main():
    server, events = await initialize()
    try:
        if args.world == 2:
            dist.init_process_group('nccl', init_method=f'tcp://{args.master}:{args.port}',
                                    rank=args.rank, world_size=2, timeout=timedelta(minutes=12), device_id=r.device)
            r.obj_pg = dist.new_group(backend='gloo', timeout=timedelta(minutes=12))
        with torch.no_grad():
            if args.rank == 1:
                await worker_models(server)
                dist.barrier()
                r.worker()
            else:
                if args.world == 2:
                    dist.barrier()
                install_head_wrappers()
                e = await execute_graph(server, spec['graph'], plan['output_ids'])
                r.record('history.json', e.history_result)
                if args.world == 2:
                    dist.broadcast_object_list([{'op': 'shutdown'}], src=0, group=r.obj_pg)
        r.record('exit.json', {'exit': 0, 'rank': args.rank, 'forward_counts': r.forward_counts,
                              'qkv': r.stage_qkv.stats})
        if args.world == 2:
            dist.destroy_process_group()
    except BaseException as error:
        r.record('failure.json', {'error': repr(error)})
        raise
    finally:
        events.close()


os.chdir(args.comfy_root)
asyncio.run(main())
