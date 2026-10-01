"""ComfyUI's normal progress hook for an isolated graph executor."""


def make_progress_hook(server, get_context, get_state, check_interrupted):
    def hook(value, total, preview_image, prompt_id=None, node_id=None):
        context = get_context()
        if context is not None:
            prompt_id = prompt_id if prompt_id is not None else context.prompt_id
            node_id = node_id if node_id is not None else context.node_id
        check_interrupted()
        if prompt_id is None:
            prompt_id = server.last_prompt_id
        if node_id is None:
            node_id = server.last_node_id
        get_state().update_progress(node_id, value, total, preview_image)
        server.send_sync('progress', {'value': value, 'max': total,
                                     'prompt_id': prompt_id, 'node': node_id}, server.client_id)
    return hook
