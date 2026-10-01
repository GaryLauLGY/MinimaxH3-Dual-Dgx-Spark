"""Exercise the actual child-process boundary without allocating a GPU."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'runtime/s02/plugin/control.py'
spec = importlib.util.spec_from_file_location('s02_environment_control', path)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class EnvironmentTests(unittest.TestCase):
    def test_configured_route_overrides_parent_and_keeps_native_allocator(self):
        for selected in ('0', '1'):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'environment.json'
                code = ('import json,os,sys; '
                        'json.dump({k:os.environ[k] for k in '
                        '["H3_S02_FIXED_CUBLAS_LMS","PYTORCH_ALLOC_CONF","PYTORCH_CUDA_ALLOC_CONF"]},'
                        'open(sys.argv[1],"w"))')
                config = dict(fabric_env={'H3_S02_FIXED_CUBLAS_LMS': selected},
                              cache_root=tmp, timeout_s=0)
                with patch.dict(os.environ, {'H3_S02_FIXED_CUBLAS_LMS': '1' if selected == '0' else '0'}), \
                        patch.object(control, 'command', return_value=[sys.executable, '-c', code, str(output)]):
                    control.run_process(config, 0, Path(tmp), {}, lambda: None)
                self.assertEqual(json.loads(output.read_text()), {
                    'H3_S02_FIXED_CUBLAS_LMS': selected, 'PYTORCH_ALLOC_CONF': 'backend:native',
                    'PYTORCH_CUDA_ALLOC_CONF': 'backend:native'})
                self.assertEqual(json.loads((Path(tmp) / 'state.json').read_text())['exit_code'], 0)


if __name__ == '__main__':
    unittest.main()
