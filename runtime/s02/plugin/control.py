"""Process ownership and loopback ComfyUI control for the dual-Spark node."""
import http.client
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request
import uuid

PACKAGE = Path(__file__).parent


def timeout_seconds(c):
    value = c.get('timeout_s', 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('timeout_s 必须为有限非负秒数；0 表示不限制任务总时长')
    return value


def config():
    data = json.loads((PACKAGE / 'config.local.json').read_text())
    if socket.gethostname() != data['expected_hostname']:
        raise RuntimeError('双机节点配置与本机名称不符')
    timeout_seconds(data)
    return data


def job_id(value):
    if str(uuid.UUID(value)) != value:
        raise ValueError('Invalid job UUID')
    return value


def request(base, path, body=None):
    if not base.startswith('http://127.0.0.1:'):
        raise ValueError('The control endpoint must use the configured loopback SSH bridge')
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)
    except http.client.HTTPException as error:
        raise ConnectionError('ComfyUI control bridge disconnected') from error


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def command(c, rank, folder, settings, refs=None):
    return [c['python'], '-u', c['backend'] + '/author_runner.py',
            '--comfy-root', c['comfy_root'], '--legacy-backend', c['legacy_backend'],
            '--rank', str(rank), '--world', str(settings['world_size']),
            '--master', c['master_addr'], '--port', str(c['master_port']),
            '--output', str(folder), '--graph-file', str(folder / 'graph.json')]


def stop_process(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_process(c, rank, folder, settings, check, refs=None, watch=None):
    timeout_s = timeout_seconds(c)
    env = dict(os.environ, **c['fabric_env'], PYTHONDONTWRITEBYTECODE='1', NCCL_IB_DISABLE='0', NCCL_DEBUG='INFO',
               TRITON_CACHE_DIR=c['cache_root'] + '/triton', CUDA_CACHE_PATH=c['cache_root'] + '/cuda')
    # ComfyUI's parent process sets cudaMallocAsync in its environment. The
    # qualified standalone dynamic-VRAM runner requires the native allocator.
    env['PYTORCH_ALLOC_CONF'] = 'backend:native'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:native'
    cmd = command(c, rank, folder, settings, refs)
    write_json(folder / 'command.json', {'argv': cmd, 'settings': settings, 'rank': rank, 'hostname': socket.gethostname(), 'timeout_s': timeout_s})
    start = time.monotonic()
    with (folder / f'rank{rank}.log').open('w') as log:
        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(folder / 'state.json', {'state': 'running', 'pid': process.pid, 'rank': rank})
        try:
            while process.poll() is None:
                check()
                if timeout_s > 0 and time.monotonic() - start > timeout_s:
                    raise TimeoutError(f'双机任务达到配置的总时长上限 {timeout_s} 秒，已停止本次子进程')
                if watch:
                    watch()
                time.sleep(0.5)
            if process.returncode:
                raise RuntimeError(f'双机 rank {rank} 失败，详见 {folder}/rank{rank}.log')
            write_json(folder / 'state.json', {'state': 'completed', 'exit_code': 0, 'rank': rank, 'wall_s': time.monotonic() - start})
        except BaseException as error:
            stop_process(process)
            write_json(folder / 'state.json', {'state': 'failed', 'error': str(error), 'rank': rank, 'wall_s': time.monotonic() - start})
            raise
