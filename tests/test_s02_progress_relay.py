import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).resolve().parents[1] / 'runtime/s02/author_backend/progress_relay.py'
spec = importlib.util.spec_from_file_location('progress_relay', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProgressTests(unittest.TestCase):
    def test_native_sampler_context_and_exact_step_values(self):
        messages, updates = [], []
        server = SimpleNamespace(client_id='browser', last_node_id='fallback', last_prompt_id='fallback',
                                 send_sync=lambda *a: messages.append(a))
        state = SimpleNamespace(update_progress=lambda *a: updates.append(a))
        context = SimpleNamespace(prompt_id='child', node_id='106')
        hook = module.make_progress_hook(server, lambda: context, lambda: state, lambda: None)
        hook(4, 10, None)
        self.assertEqual(updates, [('106', 4, 10, None)])
        self.assertEqual(messages, [('progress', {'value': 4, 'max': 10, 'prompt_id': 'child', 'node': '106'}, 'browser')])

    def test_interrupt_is_not_swallowed(self):
        def interrupted():
            raise InterruptedError('cancelled')
        hook = module.make_progress_hook(None, lambda: None, lambda: None, interrupted)
        with self.assertRaises(InterruptedError):
            hook(1, 2, None)


if __name__ == '__main__':
    unittest.main()
