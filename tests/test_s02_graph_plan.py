import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'runtime/s02/plugin/graph_plan.py'
spec = importlib.util.spec_from_file_location('graph_plan', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    def n(kind, **kw):
        return {'class_type': kind, 'inputs': kw}
    return {'69': n('UNETLoader', unet_name='Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors', weight_dtype='default'),
            '64': n('CLIPLoader', clip_name='encoder'),
            '118': n('Lora Loader Stack (rgthree)', model=['69', 0], clip=['64', 0], strength_02=0.875, lora_02='turbo'),
            '127': n('LoraLoaderModelOnly', model=['118', 0], strength_model=0.375, lora_name='lms'),
            '199': n('H3AdaLNLoRAFix', model=['127', 0], mode='port'),
            '57': n('BasicGuider', model=['118', 0], conditioning=['56', 0]),
            '126': n('BasicGuider', model=['199', 0], conditioning=['56', 0]),
            '108': n('SamplerCustomAdvanced', guider=['57', 0]),
            '128': n('SamplerCustomAdvanced', guider=['57', 0]),
            '106': n('SamplerCustomAdvanced', guider=['126', 0]),
            '65': n('VAELoader', vae_name='minimax_h3_video_vae_int8_convrot.safetensors'),
            '109': n('VAEDecode', vae=['65', 0]), '138': n('VAEDecode', vae=['65', 0]),
            '135': n('VHS_VideoCombine', images=['138', 0]), '141': n('VHS_VideoCombine', images=['109', 0])}


class PlanTests(unittest.TestCase):
    def test_native_step_progress_preserves_values_and_parent_identity(self):
        event = {'event': 'progress_state', 'data': {'prompt_id': 'child', 'nodes': {
            '106': {'node_id': '106', 'prompt_id': 'child', 'value': 4, 'max': 10, 'state': 'running'}}}}
        before = copy.deepcopy(event)
        kind, data = module.remap_child_event(event, 'parent', ['135', '141'])
        self.assertEqual(event, before)
        self.assertEqual(kind, 'progress_state')
        self.assertEqual(data['prompt_id'], 'parent')
        self.assertEqual(data['nodes']['106'], dict(before['data']['nodes']['106'], prompt_id='parent'))

    def test_preview_forwarding_does_not_complete_parent_early(self):
        event = {'event': 'executed', 'data': {'node': '135', 'prompt_id': 'child', 'output': {'gifs': ['first']}}}
        kind, data = module.remap_child_event(event, 'parent', ['135', '141'])
        self.assertEqual(data['output'], event['data']['output'])
        self.assertEqual(data['node'], '135')
        self.assertIsNone(module.remap_child_event({'event': 'execution_success', 'data': {}}, 'parent', ['135', '141']))
        self.assertIsNone(module.remap_child_event({'event': 'executed', 'data': {'node': '118'}}, 'parent', ['135', '141']))

    def test_preserves_native_parameters_and_input_graph(self):
        graph = fixture()
        before = copy.deepcopy(graph)
        plan = module.compile_plan(graph)
        self.assertEqual(graph, before)
        self.assertEqual(plan['model_graph']['118']['inputs']['strength_02'], 0.875)
        self.assertEqual(plan['model_graph']['127']['inputs']['strength_model'], 0.375)
        self.assertNotIn('64', plan['model_graph'])
        self.assertEqual(plan['guider_stages'], {'57': 'turbo', '126': 'lms'})

    def test_linked_model_parameter_is_retained(self):
        graph = fixture()
        graph['222'] = {'class_type': 'PrimitiveFloat', 'inputs': {'value': 0.625}}
        graph['127']['inputs']['strength_model'] = ['222', 0]
        plan = module.compile_plan(graph)
        self.assertEqual(plan['model_graph']['222'], graph['222'])

    def test_unsupported_model_patch_fails_instead_of_being_ignored(self):
        graph = fixture()
        graph['69']['class_type'] = 'UnknownQuantizationPatch'
        with self.assertRaises(ValueError):
            module.compile_plan(graph)

    def test_original_workflow_routing_is_untouched(self):
        data = {'prompt': fixture(), 'extra_data': {'extra_pnginfo': {'workflow': {'extra': {}}}}}
        self.assertIs(module.route_prompt(data), data)

    def test_preserves_preview_node_ids_and_original_graph(self):
        import json
        data = {'prompt': fixture(), 'client_id': 'browser',
                'extra_data': {'extra_pnginfo': {'workflow': {'extra': {'h3_dual_author': {'enabled': True}}}}}}
        before = copy.deepcopy(data)
        routed = module.route_prompt(data)
        self.assertEqual(data, before)
        self.assertEqual(set(routed['prompt']), {'__h3_author_session__', '135', '141'})
        session = routed['prompt']['__h3_author_session__']['inputs']
        self.assertEqual(json.loads(session['graph_json']), fixture())
        self.assertEqual(session['parent_prompt_id'], routed['prompt_id'])


if __name__ == '__main__':
    unittest.main()
