#!/usr/bin/env python3
"""Portable controller for an existing, pinned two-Spark ComfyUI installation.

Uses SSH from this controller to each node; no node-to-node SSH keys are needed.
Run data stays in ignored local runs/ and an explicitly configured remote scratch.
"""
import argparse
import io
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
COMFY_COMMIT = 'ee71d5c4993f29086b27fde1629a945ae48425bf'
RUNTIME_FILES = ('lab_runner.py', 'sp_primitives.py', 'sp_protocol.py', 'parallel_vae.py')
SINGULARITY_FILES = ('singularity_runner.py', 'sp_primitives.py', 'sp_protocol.py', 'parallel_vae.py', 'stage_qkv.py')
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=30']
ENV_KEYS = {'NCCL_IB_HCA', 'NCCL_IB_GID_INDEX', 'NCCL_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME'}

# Runs via the configured remote Python, stdin only; no remote file is installed.
PREFLIGHT = r'''
import hashlib, importlib.metadata, json, os, pathlib, shutil, socket, subprocess, sys, urllib.request
n = json.loads(sys.argv[1])
pin = sys.argv[2]
verify = sys.argv[3] == '1'
profile = json.loads(sys.argv[4])
assert socket.gethostname() == n['expected_hostname'], 'SSH resolved to the wrong host'
assert shutil.which('timeout'), 'GNU timeout is required'
root = pathlib.Path(n['comfy_root']).resolve()
scratch = pathlib.Path(n['scratch']).resolve()
assert root != scratch and root not in scratch.parents, 'Scratch must be outside ComfyUI'
assert scratch != root and scratch not in root.parents, 'Scratch must not contain ComfyUI'
head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
assert head == pin, 'ComfyUI commit differs from the tested pin'
subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--', 'comfy', 'comfy_extras', 'nodes.py', 'folder_paths.py'], check=True)
queues = []
for port in n['comfy_ports']:
    with urllib.request.urlopen('http://127.0.0.1:%d/queue' % port, timeout=5) as response:
        q = json.load(response)
    assert not q['queue_running'] and not q['queue_pending'], 'ComfyUI queue is busy'
    queues.append({'port': port, 'running': len(q['queue_running']), 'pending': len(q['queue_pending'])})
sys.path.insert(0, str(root))
sys.argv = [sys.argv[0]]
import folder_paths
weights = [
    ('diffusion_models', 'minimax_h3_ref2va_int8_convrot.safetensors', '9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9'),
    ('text_encoders', 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', None),
    ('vae', 'minimax_h3_video_vae_fp16.safetensors', '7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522'),
    ('vae', 'minimax_h3_audio_vae_fp32.safetensors', None),
]
if profile:
    for relative in profile['required_files']:
        assert (root / relative).is_file(), 'Missing Singularity dependency: ' + relative
    weights = [(m['category'], m['name'], m['sha256']) for m in profile['models']]
models = []
for category, name, expected in weights:
    # The upscaler's search category is registered by its custom node, which
    # this read-only preflight deliberately does not import.
    relative = next((m.get('relative_path') for m in profile.get('models', []) if m['name'] == name), None)
    path = root / relative if relative else pathlib.Path(folder_paths.get_full_path_or_raise(category, name))
    item = {'name': name, 'bytes': path.stat().st_size}
    if verify and expected:
        with path.open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
        assert digest == expected, 'Weight SHA256 mismatch: ' + name
        item['sha256'] = digest
    models.append(item)
versions = {name: importlib.metadata.version(name) for name in ['torch', 'comfy-kitchen']}
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], text=True).strip()
processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
print(json.dumps({'hostname': socket.gethostname(), 'comfy_commit': head, 'queues': queues, 'models': models, 'versions': versions, 'gpu': gpu, 'compute_processes': processes}))
'''


def run_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise ValueError('run ID must be 1-64 letters, numbers, hyphens or underscores')
    return value


def load_config(path, allow_placeholders=False):
    c = json.loads(Path(path).read_text())
    if len(c['nodes']) != 2:
        raise ValueError('exactly two nodes must be configured')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', c['master_addr']):
        raise ValueError('master_addr must be a fabric IPv4 address or hostname')
    if not 1024 <= c['master_port'] <= 65535:
        raise ValueError('master_port must be between 1024 and 65535')
    for n in c['nodes']:
        if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', n['ssh']):
            raise ValueError('ssh must be a host alias or user@host, not options')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', n['expected_hostname']):
            raise ValueError('expected_hostname must be the exact remote hostname')
        for key in ('python', 'comfy_root', 'scratch'):
            p = PurePosixPath(n[key])
            if not p.is_absolute() or '..' in p.parts or len(p.parts) < 3 or '\n' in n[key]:
                raise ValueError(key + ' must be an explicit absolute path without parent traversal')
        comfy, scratch = PurePosixPath(n['comfy_root']), PurePosixPath(n['scratch'])
        if comfy == scratch or comfy in scratch.parents or scratch in comfy.parents:
            raise ValueError('scratch and ComfyUI must be separate directories')
        if not n['comfy_ports'] or any(type(p) is not int or not 1 <= p <= 65535 for p in n['comfy_ports']):
            raise ValueError('list every ComfyUI instance in comfy_ports')
        if set(n['fabric_env']) != ENV_KEYS or any(not isinstance(v, str) or not v for v in n['fabric_env'].values()):
            raise ValueError('fabric_env must specify all four documented transport settings')
    if c['nodes'][0]['expected_hostname'] == c['nodes'][1]['expected_hostname'] or c['nodes'][0]['ssh'] == c['nodes'][1]['ssh']:
        raise ValueError('head and worker must be distinct hosts')
    if not allow_placeholders and ('EDIT_' in json.dumps(c) or c['master_addr'].startswith('192.0.2.')):
        raise ValueError('edit config.example.json placeholders before contacting machines')
    return c


def ssh(node, args, **kwargs):
    return subprocess.run(['ssh', *SSH_OPTIONS, node['ssh'], shlex.join(args)], check=True, **kwargs)


def preflight(config, world, verify_weights=False, workflow='ref2va'):
    profile = json.loads((ROOT / 'runtime/singularity/profile.json').read_text()) if workflow == 'singularity' else {}
    reports = []
    for n in config['nodes'][:world]:
        try:
            result = ssh(n, ['env', 'PYTHONDONTWRITEBYTECODE=1', n['python'], '-', json.dumps(n), COMFY_COMMIT, str(int(verify_weights)), json.dumps(profile)],
                         input=PREFLIGHT, text=True, capture_output=True, timeout=900)
        except subprocess.CalledProcessError as error:
            raise RuntimeError('Preflight failed on %s:\n%s' % (n['expected_hostname'], error.stderr)) from error
        # Dependencies can log before the final structured report.
        report = json.loads(result.stdout.strip().splitlines()[-1])
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    return reports


def command_for(config, rank, args):
    n = config['nodes'][rank]
    base = str(PurePosixPath(n['scratch']) / 'runs' / run_id(args.run))
    env = {'PYTHONDONTWRITEBYTECODE': '1', 'NCCL_DEBUG': 'INFO', 'NCCL_IB_DISABLE': '0', **n['fabric_env'],
           'TRITON_CACHE_DIR': n['scratch'] + '/cache/triton', 'CUDA_CACHE_PATH': n['scratch'] + '/cache/cuda',
           'TMPDIR': base + '/tmp'}
    singularity = args.workflow == 'singularity'
    width = args.width if args.width is not None else (768 if singularity else 832)
    height = args.height if args.height is not None else (448 if singularity else 480)
    runner = 'singularity_runner.py' if singularity else 'lab_runner.py'
    command = ['env', *[k + '=' + v for k, v in env.items()], 'timeout', '--signal=TERM', '--kill-after=30s', str(args.timeout),
               n['python'], '-u', base + '/runtime/' + runner, '--rank', str(rank), '--world', str(args.world),
               '--master', config['master_addr'], '--port', str(config['master_port']), '--comfy-root', n['comfy_root'], '--output', base + '/output',
               '--width', str(width), '--height', str(height), '--frames', str(args.frames),
               '--seed', str(args.seed), '--repeats', str(args.repeats), '--exchange', args.exchange,
               '--attention-chunks', str(args.attention_chunks), '--gather-chunks', str(args.gather_chunks)]
    if singularity:
        if args.keep_stage_qkv:
            command.append('--keep-stage-qkv')
        for name in ('reference_video', 'compare_dir'):
            if getattr(args, name) is not None:
                command.extend(['--' + name.replace('_', '-'), getattr(args, name)])
        if args.compare_run_zero:
            command.append('--compare-run-zero')
    else:
        command.extend(['--mode', args.mode, '--steps', str(args.steps if args.steps is not None else 20), '--benchmark-shapes', args.benchmark_shapes])
    if args.parallel_vae:
        command.append('--parallel-vae')
    if args.generate_after_benchmark:
        command.append('--generate-after-benchmark')
    if args.prompt is not None:
        command.extend(['--prompt', args.prompt])
    return base, command


def stage(node, base, workflow='ref2va'):
    # An existing run is never overwritten. Only this backend's known files enter the tar.
    ssh(node, ['mkdir', '-p', '--', node['scratch'] + '/runs', node['scratch'] + '/cache/triton', node['scratch'] + '/cache/cuda'])
    ssh(node, ['mkdir', '--', base])
    ssh(node, ['mkdir', '--', base + '/runtime', base + '/tmp'])
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w') as tar:
        folder = ROOT / 'runtime' / 'singularity' if workflow == 'singularity' else ROOT / 'runtime'
        for name in SINGULARITY_FILES if workflow == 'singularity' else RUNTIME_FILES:
            tar.add(folder / name, arcname=name, recursive=False)
    ssh(node, ['tar', '-xf', '-', '-C', base + '/runtime'], input=archive.getvalue())


def launch(config, args):
    plan = [(rank, *command_for(config, rank, args)) for rank in range(args.world)]
    if args.dry_run:
        for rank, base, command in plan:
            print(json.dumps({'rank': rank, 'ssh': config['nodes'][rank]['ssh'], 'new_remote_run': base, 'command': command}, indent=2))
        return 0
    preflight(config, args.world, args.verify_weights, args.workflow)
    local = ROOT / 'runs' / args.run
    local.mkdir(parents=True, exist_ok=False)
    (local / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    for rank, base, _ in plan:
        stage(config['nodes'][rank], base, args.workflow)
    # Recheck queues after staging and immediately before GPU work starts.
    preflight(config, args.world, workflow=args.workflow)
    start = time.monotonic()
    procs, files, codes = [], [], {}
    try:
        for rank, _, command in reversed(plan):
            f = (local / ('rank%d.log' % rank)).open('w')
            files.append(f)
            process = subprocess.Popen(['ssh', *SSH_OPTIONS, config['nodes'][rank]['ssh'], shlex.join(command)], stdout=f, stderr=subprocess.STDOUT)
            procs.append((rank, process))
            print('Started rank %d; log: %s' % (rank, f.name), flush=True)
        for rank, process in procs:
            codes[str(rank)] = process.wait()
        return 0 if all(code == 0 for code in codes.values()) else 1
    finally:
        for rank, process in procs:
            if process.poll() is None:
                process.terminate()
                print('SSH interrupted for rank %d. The remote timeout remains the maximum job lifetime; inspect the configured host before starting another run.' % rank, flush=True)
            else:
                codes.setdefault(str(rank), process.returncode)
        for f in files:
            f.close()
        (local / 'status.json').write_text(json.dumps({'exit_codes': codes, 'all_ranks_finished': len(codes) == args.world,
                                                     'wall_s': time.monotonic() - start}, indent=2) + '\n')


def collect(config, args):
    node = config['nodes'][0]
    remote = str(PurePosixPath(node['scratch']) / 'runs' / run_id(args.run) / 'output')
    # rsync protects remote arguments, and a new local directory prevents overwrite.
    destination = ROOT / 'runs' / args.run / 'collected'
    destination.mkdir(parents=True, exist_ok=False)
    subprocess.run(['rsync', '-a', '--protect-args', '-e', shlex.join(['ssh', *SSH_OPTIONS]),
                    node['ssh'] + ':' + remote + '/', str(destination) + '/'], check=True)
    print(destination)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['validate', 'run', 'status', 'collect'])
    p.add_argument('--config', default=str(ROOT / 'config.local.json'))
    p.add_argument('--run', type=run_id)
    p.add_argument('--workflow', choices=['ref2va', 'singularity'], default='ref2va')
    p.add_argument('--mode', choices=['link', 'parity', 'benchmark', 'generate'], default='generate')
    p.add_argument('--world', type=int, choices=[1, 2], default=2)
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--verify-weights', action='store_true')
    for flag, default in [('width', None), ('height', None), ('frames', 56), ('steps', None), ('repeats', 4), ('seed', 20260928), ('attention-chunks', 4), ('gather-chunks', 4)]:
        p.add_argument('--' + flag, type=int, default=default)
    p.add_argument('--exchange', choices=['alltoall', 'allgather'], default='allgather')
    p.add_argument('--benchmark-shapes', default='832x480x56')
    p.add_argument('--parallel-vae', action='store_true')
    p.add_argument('--generate-after-benchmark', action='store_true')
    p.add_argument('--prompt')
    p.add_argument('--keep-stage-qkv', action='store_true', help='Singularity only: retain separate Turbo/LMS effective QKV caches')
    p.add_argument('--reference-video', help='Singularity only: video on the head; also supplies its first frame as a reference image')
    p.add_argument('--compare-dir', help='Singularity only: baseline output directory on the head')
    p.add_argument('--compare-run-zero', action='store_true')
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if a.action != 'validate' and not a.run:
        p.error('--run is required')
    if a.dry_run and a.action != 'run':
        p.error('--dry-run is supported by run only')
    if a.action == 'run':
        if a.workflow == 'singularity':
            if a.mode != 'generate' or a.generate_after_benchmark or a.steps is not None:
                p.error('Singularity uses fixed 2+10 steps; only --mode generate is supported')
        elif a.keep_stage_qkv or a.reference_video or a.compare_dir or a.compare_run_zero:
            p.error('cache/reference/comparison options require --workflow singularity')
        if a.compare_run_zero and not a.compare_dir:
            p.error('--compare-run-zero requires --compare-dir')
        for value in (a.reference_video, a.compare_dir):
            if value is not None and (not PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts or '\n' in value):
                p.error('reference and comparison paths must be absolute paths on the head without traversal')
        if a.mode != 'generate' and a.world != 2:
            p.error('diagnostics require --world 2')
        if a.parallel_vae and a.world != 2:
            p.error('--parallel-vae requires --world 2')
        if a.generate_after_benchmark and a.mode != 'benchmark':
            p.error('--generate-after-benchmark requires --mode benchmark')
        if min(v for v in (a.timeout, a.width, a.height, a.frames, a.steps, a.repeats, a.attention_chunks, a.gather_chunks) if v is not None) <= 0:
            p.error('time, dimensions, repeats and chunk counts must be positive')
        if any(v is not None and v % 16 for v in (a.width, a.height)):
            p.error('dimensions must be multiples of 16')
        if not re.fullmatch(r'[1-9][0-9]*x[1-9][0-9]*x[1-9][0-9]*(,[1-9][0-9]*x[1-9][0-9]*x[1-9][0-9]*)*', a.benchmark_shapes):
            p.error('invalid benchmark-shapes')
        if any(int(v) % 16 for shape in a.benchmark_shapes.split(',') for v in shape.split('x')[:2]):
            p.error('benchmark width and height must be multiples of 16')
    if a.action == 'status':
        path = ROOT / 'runs' / a.run / 'status.json'
        print(path.read_text() if path.exists() else 'No final status. Inspect rank logs; this is not a remote liveness check.')
        return 0
    c = load_config(a.config, allow_placeholders=a.dry_run)
    if a.action == 'validate':
        preflight(c, a.world, a.verify_weights, a.workflow)
    elif a.action == 'run':
        return launch(c, a)
    else:
        collect(c, a)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error))
