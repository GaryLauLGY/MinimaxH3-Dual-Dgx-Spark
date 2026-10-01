"""Opt-in execution bridge for the author's unmodified ComfyUI node graph."""
import fcntl
import gc
import json
import logging
from pathlib import Path
import socket
import time
import uuid

from aiohttp import web
import comfy.model_management as mm
from server import PromptServer
from . import control
from .graph_plan import compile_plan, route_prompt, remap_child_event


def unload():
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()


def create_run(c, job):
    p = Path(c['runs_root']) / ('author_' + control.job_id(job))
    p.mkdir(parents=True, exist_ok=False)
    return p


@PromptServer.instance.routes.get('/h3-author/info')
async def info(request):
    c = control.config()
    return web.json_response({**{k: c[k] for k in ('role', 'expected_hostname', 'package_id')},
                              'timeout_s': control.timeout_seconds(c), 'timeout_policy': 'optional_wall_clock_v1'})


@PromptServer.instance.routes.get('/h3-author/state/{job}')
async def state(request):
    try:
        job = control.job_id(request.match_info['job'])
    except ValueError:
        return web.json_response({'error': 'Invalid UUID'}, status=400)
    p = Path(control.config()['runs_root']) / ('author_' + job) / 'state.json'
    return web.json_response(json.loads(p.read_text()) if p.is_file() else {'state': 'waiting'})


class H3AuthorWorker:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'job': ('STRING',), 'profile_json': ('STRING',)}}
    RETURN_TYPES = ('STRING',)
    FUNCTION = 'run'
    OUTPUT_NODE = True
    CATEGORY = '双DGX/内部协作'

    def run(self, job, profile_json):
        c = control.config()
        if c['role'] != 'worker':
            raise RuntimeError('请仅在DGX2运行协作节点')
        p = create_run(c, job)
        control.write_json(p / 'graph.json', json.loads(profile_json))
        with (Path(c['runs_root']) / '.gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unload()
            control.run_process(c, 1, p, {'world_size': 2}, mm.throw_exception_if_processing_interrupted)
        return (job,)


class H3AuthorGraph:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'graph_json': ('STRING',), 'world_size': ('INT', {'default': 2, 'min': 1, 'max': 2}),
                             'client_id': ('STRING',), 'parent_prompt_id': ('STRING',), 'validation_dir': ('STRING',)},
                'hidden': {'extra_pnginfo': 'EXTRA_PNGINFO'}}
    RETURN_TYPES = ('H3_AUTHOR_RESULT',)
    FUNCTION = 'run'
    CATEGORY = '双DGX/内部协作'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def run(self, graph_json, world_size, client_id, parent_prompt_id, validation_dir, extra_pnginfo=None):
        c = control.config()
        if c['role'] != 'head' or world_size not in (1, 2):
            raise ValueError('作者双机图请在DGX1运行')
        graph = json.loads(graph_json)
        plan = compile_plan(graph)
        if validation_dir:
            base = Path(validation_dir).resolve()
            if not base.is_relative_to(Path(c['runs_root']).resolve()) or not base.is_dir():
                raise ValueError('验证目录须为本项目已经存在的run')
        peer = c['worker_url']
        if world_size == 2:
            expected = {'role': 'worker', 'expected_hostname': c['worker_hostname'], 'package_id': c['package_id'],
                        'timeout_s': control.timeout_seconds(c), 'timeout_policy': 'optional_wall_clock_v1'}
            if control.request(peer, '/h3-author/info') != expected:
                raise RuntimeError('DGX2作者图节点版本、身份或总时长限制不一致')
            if any(control.request(peer, '/queue').values()):
                raise RuntimeError('DGX2有任务，等待其空闲后再运行作者双机图')
        job = str(uuid.uuid4())
        p = create_run(c, job)
        spec = dict(graph=graph, plan=plan, extra_pnginfo=extra_pnginfo or {}, job=job,
                    evidence=bool((extra_pnginfo or {}).get('workflow', {}).get('extra', {}).get('h3_dual_author', {}).get('evidence')),
                    validation_dir=validation_dir)
        control.write_json(p / 'graph.json', spec)
        submitted = False
        with (Path(c['runs_root']) / '.gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                if world_size == 2:
                    submitted = True
                    worker = {'1': {'class_type': 'H3AuthorWorker', 'inputs': {
                        'job': job, 'profile_json': json.dumps({'plan': plan, 'job': job})}}}
                    result = control.request(peer, '/prompt', {'prompt_id': job, 'prompt': worker, 'client_id': 'h3-author-' + job})
                    if result.get('prompt_id') != job or result.get('node_errors'):
                        raise RuntimeError('DGX2未接受作者图协作任务')
                    control.write_json(p / 'peer.json', result)
                    deadline = time.monotonic() + 90
                    while control.request(peer, '/h3-author/state/' + job)['state'] != 'running':
                        mm.throw_exception_if_processing_interrupted()
                        if control.request(peer, '/history/' + job) or time.monotonic() > deadline:
                            raise RuntimeError('DGX2作者图协作启动失败')
                        time.sleep(0.5)
                unload()
                offset, last = [0], [0.0]
                def watch():
                    # Forward actual original-node progress, never a made-up percentage.
                    events = p / 'events.jsonl'
                    if events.is_file():
                        with events.open('rb') as f:
                            f.seek(offset[0])
                            for line in f:
                                if not line.endswith(b'\n'):
                                    break
                                try:
                                    event = json.loads(line)
                                except json.JSONDecodeError:
                                    break
                                forwarded = remap_child_event(event, parent_prompt_id, plan['output_ids'])
                                if forwarded is not None:
                                    PromptServer.instance.send_sync(*forwarded, client_id or None)
                                offset[0] += len(line)
                    if world_size == 2 and time.monotonic() - last[0] > 3:
                        last[0] = time.monotonic()
                        if control.request(peer, '/h3-author/state/' + job)['state'] == 'failed':
                            raise RuntimeError('DGX2作者图计算失败')
                control.run_process(c, 0, p, {'world_size': world_size}, mm.throw_exception_if_processing_interrupted, watch=watch)
                if world_size == 2:
                    deadline = time.monotonic() + 30
                    while True:
                        h = control.request(peer, '/history/' + job)
                        if job in h:
                            control.write_json(p / 'worker_history.json', h[job])
                            if h[job]['status']['status_str'] != 'success':
                                raise RuntimeError('DGX2返回失败')
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError('DGX2完成确认超时')
                        time.sleep(0.5)
                    submitted = False
            finally:
                if submitted:
                    try:
                        control.request(peer, '/api/jobs/' + job + '/cancel', {})
                    except OSError:
                        logging.exception('Cannot cancel owned author worker %s', job)
        result = json.loads((p / 'history.json').read_text())['outputs']
        result['_run_dir'] = str(p)
        return (result,)


class H3AuthorOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'result': ('H3_AUTHOR_RESULT',), 'original_id': ('STRING',)}}
    RETURN_TYPES = ()
    FUNCTION = 'run'
    OUTPUT_NODE = True
    CATEGORY = '双DGX/内部协作'

    def run(self, result, original_id):
        return {'ui': result[original_id]}


NODE_CLASS_MAPPINGS = {'H3AuthorGraph': H3AuthorGraph, 'H3AuthorWorker': H3AuthorWorker, 'H3AuthorOutput': H3AuthorOutput}
PromptServer.instance.add_on_prompt_handler(route_prompt)
