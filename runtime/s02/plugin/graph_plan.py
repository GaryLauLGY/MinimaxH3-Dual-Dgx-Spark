"""Inspect the submitted native graph without evaluating or replacing its parameters."""
import copy

MODEL_NODES = {'UNETLoader', 'ModelAttentionBackend', 'BlockSparseAttention',
               'MiniMaxChunkFeedForward', 'LoraLoaderModelOnly', 'Lora Loader Stack (rgthree)',
               'H3AdaLNLoRAFix'}


def remap_child_event(event, parent_prompt_id, output_ids):
    """Relay native progress/finished previews without child job identity leaks."""
    kind = event['event']
    if kind not in ('executing', 'progress', 'progress_state', 'executed'):
        return None
    data = copy.deepcopy(event['data'])
    if kind == 'executed' and data.get('node') not in output_ids:
        return None
    data['prompt_id'] = parent_prompt_id
    if kind == 'progress_state':
        for state in data.get('nodes', {}).values():
            state['prompt_id'] = parent_prompt_id
    return kind, data


def is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


def compile_plan(prompt):
    graph = copy.deepcopy(prompt)
    guiders = {k: n for k, n in graph.items() if n['class_type'] == 'BasicGuider'}
    samplers = {k: n for k, n in graph.items() if n['class_type'] == 'SamplerCustomAdvanced'}
    if len(guiders) != 2 or len(samplers) != 3:
        raise ValueError('当前作者图双机入口需要原版两个BasicGuider和三个SamplerCustomAdvanced')
    model_graph = {}
    visiting = set()

    def visit(key, model=False):
        if key in model_graph:
            return
        if key in visiting:
            raise ValueError('Model graph contains a cycle')
        visiting.add(key)
        n = copy.deepcopy(graph[key])
        if model and n['class_type'] not in MODEL_NODES:
            raise ValueError('尚未验证的模型补丁: ' + n['class_type'])
        if n['class_type'] == 'Lora Loader Stack (rgthree)':
            n['inputs'].pop('clip', None)  # No text encoder needed by the compute worker.
        for name, value in n['inputs'].items():
            if is_link(value):
                visit(value[0], name == 'model')
        model_graph[key] = n
        visiting.remove(key)

    origins = {gid: n['inputs']['model'][0] for gid, n in guiders.items()}
    # Identify the LMS stage by its actual upstream AdaLN node, not UI positions.
    lms = [gid for gid, model_id in origins.items() if graph[model_id]['class_type'] == 'H3AdaLNLoRAFix']
    if len(lms) != 1:
        raise ValueError('需要一个原版LMS AdaLN输出作为第二遍模型')
    stages = {gid: ('lms' if gid == lms[0] else 'turbo') for gid in guiders}
    for mid in origins.values():
        visit(mid, True)
    unets = [n for n in model_graph.values() if n['class_type'] == 'UNETLoader']
    if len(unets) != 1 or unets[0]['inputs']['unet_name'] != 'Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors':
        raise ValueError('当前仅验证Singularity v1.3 INT8同一个基础模型')
    vaes = {k: graph[n['inputs']['vae'][0]]['inputs']['vae_name']
            for k, n in graph.items() if n['class_type'] == 'VAEDecode'}
    if len(vaes) != 2 or set(vaes.values()) != {'minimax_h3_video_vae_int8_convrot.safetensors'}:
        raise ValueError('当前并行VAE需要原版两个视频VAEDecode及INT8视频VAE')
    outputs = [k for k, n in graph.items() if n['class_type'] == 'VHS_VideoCombine']
    if len(outputs) != 2:
        raise ValueError('请保留原作者两遍VHS视频输出')
    return dict(model_graph=model_graph,
                model_outputs={stages[g]: origins[g] for g in guiders},
                guider_stages=stages, sampler_ids=list(samplers), vae_ids=list(vaes), output_ids=outputs)


def route_prompt(data):
    """Only opt-in workflow copies are routed; the original T01 stays native."""
    extra = data.get('extra_data', {}).get('extra_pnginfo', {})
    mode = extra.get('workflow', {}).get('extra', {}).get('h3_dual_author', {})
    if not mode.get('enabled'):
        return data
    graph = data['prompt']
    if any(n.get('class_type') == 'H3AuthorGraph' for n in graph.values()):
        return data
    import json
    import uuid
    # Validation errors must not silently fall back to single GPU.
    # The executor node performs this check inside the owned ComfyUI job.
    outputs = [k for k, n in graph.items() if n.get('class_type') == 'VHS_VideoCombine']
    parent_id = data.get('prompt_id') or str(uuid.uuid4())
    job_node = '__h3_author_session__'
    routed = {job_node: {'class_type': 'H3AuthorGraph', 'inputs': {
        'graph_json': json.dumps(graph, ensure_ascii=False),
        'world_size': int(mode.get('world_size', 2)),
        'client_id': data.get('client_id', ''),
        'parent_prompt_id': parent_id,
        'validation_dir': mode.get('validation_dir', ''),
    }}}
    for key in outputs:
        routed[key] = {'class_type': 'H3AuthorOutput', 'inputs': {'result': [job_node, 0], 'original_id': key}}
    if not outputs:
        routed['__h3_author_error__'] = {'class_type': 'H3AuthorOutput', 'inputs': {'result': [job_node, 0], 'original_id': ''}}
    result = dict(data, prompt=routed, prompt_id=parent_id)
    result.pop('partial_execution_targets', None)
    return result
