import contextlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import cluster
import verify_singularity


class SingularityTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / 'config.example.json').read_text())

    def test_workflow_commands_keep_distinct_defaults(self):
        for workflow, runner, dims in [('ref2va', 'lab_runner.py', ('832', '480')), ('singularity', 'singularity_runner.py', ('768', '448'))]:
            args = cluster.parser().parse_args(['run', '--run', 'test', '--workflow', workflow])
            _, cmd = cluster.command_for(self.config(), 0, args)
            self.assertTrue(any(s.endswith('/' + runner) for s in cmd))
            self.assertEqual((cmd[cmd.index('--width') + 1], cmd[cmd.index('--height') + 1]), dims)
            self.assertEqual('--steps' in cmd, workflow == 'ref2va')
            self.assertEqual('--mode' in cmd, workflow == 'ref2va')

    def test_singularity_dry_run_has_no_side_effects(self):
        with patch.object(cluster, 'preflight', side_effect=AssertionError('network')), patch.object(cluster, 'stage', side_effect=AssertionError('write')), patch.object(Path, 'mkdir', side_effect=AssertionError('write')), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cluster.main(['run', '--workflow', 'singularity', '--run', 'dry', '--config', str(ROOT / 'config.example.json'), '--dry-run', '--parallel-vae', '--keep-stage-qkv']), 0)
        self.assertIn('singularity_runner.py', output.getvalue())

    def test_wrong_workflow_options_fail_before_contact(self):
        invalid = [['--workflow', 'singularity', '--steps', '8'], ['--workflow', 'singularity', '--mode', 'parity'], ['--keep-stage-qkv'], ['--workflow', 'singularity', '--compare-run-zero'], ['--workflow', 'singularity', '--reference-video', '../input.mp4'], ['--workflow', 'singularity', '--width', '0'], ['--workflow', 'singularity', '--world', '1', '--parallel-vae']]
        for options in invalid:
            with self.subTest(options=options), patch.object(cluster, 'preflight', side_effect=AssertionError('network')), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cluster.main(['run', '--run', 'bad', *options])

    def test_reference_paths_and_prompt_are_single_arguments(self):
        path = '/data/reference with spaces/boat.mp4'
        prompt = 'lake; $(no-command) `no-command`\nsecond line'
        args = cluster.parser().parse_args(['run', '--run', 'quoted', '--workflow', 'singularity', '--reference-video', path, '--prompt', prompt, '--compare-dir', '/data/baseline', '--compare-run-zero', '--keep-stage-qkv'])
        _, cmd = cluster.command_for(self.config(), 0, args)
        self.assertEqual(shlex.split(shlex.join(cmd)), cmd)
        self.assertEqual(cmd[cmd.index('--reference-video') + 1], path)
        self.assertEqual(cmd[cmd.index('--prompt') + 1], prompt)

    def test_staged_files_are_backend_specific(self):
        for workflow, expected in [('ref2va', cluster.RUNTIME_FILES), ('singularity', cluster.SINGULARITY_FILES)]:
            with patch.object(cluster, 'ssh') as ssh:
                cluster.stage(self.config()['nodes'][0], '/data/scratch/runs/new', workflow)
            payload = ssh.call_args.kwargs['input']
            with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
                self.assertEqual(set(archive.getnames()), set(expected))
                self.assertNotIn('launch.py', archive.getnames())

    def test_preflight_selects_singularity_weights(self):
        with patch.object(cluster, 'ssh', return_value=subprocess.CompletedProcess([], 0, stdout='{}\n')) as ssh, contextlib.redirect_stdout(io.StringIO()):
            cluster.preflight(self.config(), 1, True, 'singularity')
        profile = json.loads(ssh.call_args.args[1][-1])
        self.assertEqual(len(profile['models']), 8)
        self.assertEqual(len(profile['required_files']), 3)
        self.assertIn('Singularity', profile['models'][0]['name'])
        self.assertTrue(all(len(m['sha256']) == 64 for m in profile['models']))

    def test_runner_help_and_validation_need_no_gpu(self):
        runner = str(ROOT / 'runtime/singularity/singularity_runner.py')
        result = subprocess.run([sys.executable, runner, '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--keep-stage-qkv', result.stdout)
        result = subprocess.run([sys.executable, runner, '--comfy-root', '/no-comfy', '--output', '/no-output', '--rank', '2', '--world', '2'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('rank must', result.stderr)
        self.assertNotIn('ModuleNotFoundError', result.stderr)

    def test_archived_benchmarks_and_media_hashes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            verify_singularity.verify()


if __name__ == '__main__':
    unittest.main()
