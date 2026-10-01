import asyncio
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


BASE = Path(__file__).resolve().parents[1] / 'runtime/s02'


def load_control(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUTHOR = load_control(BASE / 'plugin/control.py', 'author_control_test')


class TimeoutTests:
    def run_child(self, folder, config, command, check=lambda: None, watch=None):
        # Simulate twelve hours passing without waiting or allocating a GPU.
        clock = iter([0])
        with patch.object(self.control, 'command', return_value=command), \
                patch.object(self.control.time, 'monotonic', side_effect=lambda: next(clock, 43200)):
            self.control.run_process(config, 0, folder, {}, check, watch=watch)

    def test_disabled_or_missing_limit_survives_twelve_hours(self):
        for extra in ({}, {'timeout_s': 0}):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                watch = MagicMock()
                self.run_child(folder, dict(fabric_env={}, cache_root=tmp, **extra),
                               [sys.executable, '-c', 'import time; time.sleep(0.05)'], watch=watch)
                state = json.loads((folder / 'state.json').read_text())
                self.assertEqual(state['state'], 'completed')
                self.assertEqual(state['wall_s'], 43200)
                self.assertEqual(json.loads((folder / 'command.json').read_text())['timeout_s'], 0)
                watch.assert_called()

    def test_explicit_limit_still_stops_only_owned_child(self):
        other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                with self.assertRaisesRegex(TimeoutError, '10800'):
                    self.run_child(folder, dict(fabric_env={}, cache_root=tmp, timeout_s=10800),
                                   [sys.executable, '-c', 'import time; time.sleep(30)'])
                self.assertEqual(json.loads((folder / 'state.json').read_text())['state'], 'failed')
                self.assertIsNone(other.poll())
        finally:
            self.control.stop_process(other)

    def test_cancel_still_cleans_up_when_limit_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            def cancel():
                raise RuntimeError('manual cancellation')
            with self.assertRaisesRegex(RuntimeError, 'manual cancellation'), \
                    patch.object(self.control, 'stop_process', wraps=self.control.stop_process) as stop:
                self.run_child(Path(tmp), dict(fabric_env={}, cache_root=tmp, timeout_s=0),
                               [sys.executable, '-c', 'import time; time.sleep(30)'], check=cancel)
            stop.assert_called_once()
            self.assertIsNotNone(stop.call_args.args[0].poll())
            self.assertEqual(json.loads((Path(tmp) / 'state.json').read_text())['error'], 'manual cancellation')

    def test_child_error_is_not_hidden_by_disabled_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'rank 0'):
                self.run_child(Path(tmp), dict(fabric_env={}, cache_root=tmp, timeout_s=0),
                               [sys.executable, '-c', 'raise SystemExit(7)'])
            self.assertEqual(json.loads((Path(tmp) / 'state.json').read_text())['state'], 'failed')

    def test_bad_configuration_fails_before_launch(self):
        for value in (-1, None, True, '0', float('nan'), float('inf')):
            with self.subTest(value=value), patch.object(self.control.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'timeout_s'):
                    self.control.run_process({'timeout_s': value}, 0, Path('/unused'), {}, lambda: None)
                launch.assert_not_called()


class AuthorTimeoutTests(TimeoutTests, unittest.TestCase):
    control = AUTHOR


class AuthorPreflightTests(unittest.TestCase):
    def setUp(self):
        names = ['comfy', 'comfy.model_management', 'server', 'aiohttp']
        mocks = {name: MagicMock() for name in names}
        mocks['server'].PromptServer.instance.routes.get.return_value = lambda function: function
        name = 'author_timeout_node_test'
        spec = importlib.util.spec_from_file_location(name, BASE / 'plugin/__init__.py',
                                                     submodule_search_locations=[str(BASE / 'plugin')])
        self.node = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {**mocks, name: self.node}):
            spec.loader.exec_module(self.node)
        self.config = dict(role='head', worker_url='http://127.0.0.1:18193', worker_hostname='worker',
                           expected_hostname='head', package_id='new', timeout_s=0)

    def test_info_exposes_loaded_timeout_policy(self):
        with patch.object(self.node.control, 'config', return_value=self.config), \
                patch.object(self.node.web, 'json_response', side_effect=lambda value: value):
            value = asyncio.run(self.node.info(None))
        self.assertEqual(value['timeout_s'], 0)
        self.assertEqual(value['timeout_policy'], 'optional_wall_clock_v1')

    def test_old_or_different_worker_policy_blocks_before_gpu(self):
        base = dict(role='worker', expected_hostname='worker', package_id='new')
        for info in (base, dict(base, timeout_s=10800, timeout_policy='optional_wall_clock_v1')):
            with self.subTest(info=info), patch.object(self.node.control, 'config', return_value=self.config), \
                    patch.object(self.node, 'compile_plan', return_value={}), \
                    patch.object(self.node.control, 'request', return_value=info), \
                    patch.object(self.node, 'create_run') as create, patch.object(self.node, 'unload') as unload:
                with self.assertRaisesRegex(RuntimeError, '总时长限制'):
                    self.node.H3AuthorGraph().run('{}', 2, '', '', '')
                create.assert_not_called()
                unload.assert_not_called()

    def test_matching_policy_passes_preflight(self):
        peer = dict(role='worker', expected_hostname='worker', package_id='new', timeout_s=0,
                    timeout_policy='optional_wall_clock_v1')
        with patch.object(self.node.control, 'config', return_value=self.config), \
                patch.object(self.node, 'compile_plan', return_value={}), \
                patch.object(self.node.control, 'request', side_effect=[peer, {'queue_running': [], 'queue_pending': []}]), \
                patch.object(self.node, 'create_run', side_effect=RuntimeError('preflight passed')):
            with self.assertRaisesRegex(RuntimeError, 'preflight passed'):
                self.node.H3AuthorGraph().run('{}', 2, '', '', '')


if __name__ == '__main__':
    unittest.main()
