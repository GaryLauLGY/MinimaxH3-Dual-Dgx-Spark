import contextlib
import copy
import io
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import cluster
import verify_results


class ControllerTests(unittest.TestCase):
    def config(self, change=None):
        c = json.loads((ROOT / 'config.example.json').read_text())
        if change:
            change(c)
        return c

    def load(self, c, placeholders=True):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(c))
            return cluster.load_config(path, allow_placeholders=placeholders)

    def test_example_requires_editing_before_live_use(self):
        with self.assertRaisesRegex(ValueError, 'placeholders'):
            self.load(self.config(), placeholders=False)

    def test_distinct_hosts_required(self):
        c = self.config()
        c['nodes'][1] = copy.deepcopy(c['nodes'][0])
        with self.assertRaisesRegex(ValueError, 'distinct'):
            self.load(c)

    def test_scratch_cannot_touch_comfy(self):
        for scratch in ['/home/EDIT_USER/ComfyUI', '/home/EDIT_USER/ComfyUI/tmp', '/home/EDIT_USER']:
            with self.subTest(scratch=scratch):
                c = self.config()
                c['nodes'][0]['scratch'] = scratch
                with self.assertRaisesRegex(ValueError, 'separate'):
                    self.load(c)

    def test_ssh_option_injection_rejected(self):
        c = self.config()
        c['nodes'][0]['ssh'] = '-oProxyCommand=bad'
        with self.assertRaisesRegex(ValueError, 'ssh'):
            self.load(c)

    def test_traversal_and_shell_run_ids_rejected(self):
        for value in ['../other', '/absolute', 'x;id', 'a b', 'a\nb', '']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                cluster.run_id(value)

    def test_dry_run_never_contacts_or_writes(self):
        with patch.object(cluster, 'preflight', side_effect=AssertionError('network')), \
             patch.object(cluster, 'stage', side_effect=AssertionError('remote write')), \
             patch.object(cluster.subprocess, 'Popen', side_effect=AssertionError('process')), \
             patch.object(Path, 'mkdir', side_effect=AssertionError('local write')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cluster.main(['run', '--config', str(ROOT / 'config.example.json'), '--run', 'dry-check', '--dry-run']), 0)
        self.assertIn('rank', output.getvalue())

    def test_prompt_is_one_remote_argument(self):
        prompt = 'sailboat; $(do-not-execute) `not-a-command` "quote"\nsecond line'
        args = cluster.parser().parse_args(['run', '--run', 'quoted', '--prompt', prompt])
        _, command = cluster.command_for(self.config(), 0, args)
        self.assertEqual(shlex.split(shlex.join(command)), command)
        self.assertEqual(command[-1], prompt)

    def test_failed_preflight_blocks_all_staging(self):
        args = cluster.parser().parse_args(['run', '--run', 'blocked'])
        with patch.object(cluster, 'preflight', side_effect=RuntimeError('queue busy')), \
             patch.object(cluster, 'stage', side_effect=AssertionError('must not upload')), \
             patch.object(Path, 'mkdir', side_effect=AssertionError('must not write')), \
             self.assertRaisesRegex(RuntimeError, 'queue busy'):
            cluster.launch(self.config(), args)

    def test_invalid_single_node_diagnostic_fails_before_ssh(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cluster.main(['run', '--run', 'bad', '--world', '1', '--mode', 'benchmark'])

    def test_runtime_help_requires_no_gpu_dependencies(self):
        result = cluster.subprocess.run([sys.executable, str(ROOT / 'runtime/lab_runner.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--parallel-vae', result.stdout)

    def test_published_results(self):
        with contextlib.redirect_stdout(io.StringIO()):
            verify_results.verify()


if __name__ == '__main__':
    unittest.main()
