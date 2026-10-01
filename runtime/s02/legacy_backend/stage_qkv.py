"""Own only effective QKV row copies for the currently active LoRA stage.

Resolve weights through the native dynamic-VRAM caster before slicing. Never
copy raw unpatched checkpoint weights, nor retain pointers into evictable VBAR.
"""
import copy
import dataclasses
from types import MethodType
import torch
import comfy.ops
import comfy.model_patcher
import comfy.model_prefetch
from comfy.quant_ops import QuantizedTensor
from comfy.weight_adapter.lora import LoRAAdapter

_stage = None
_cache = {}
retain_stages = False
stats = {'builds': 0, 'hits': 0, 'stage_switches': 0, 'owned_bytes': 0,
         'prefetch_skips': 0, 'rebuild_reasons': {}}


def activate(stage):
    global _stage
    if stage != _stage:
        if not retain_stages:
            _cache.clear()
            stats['owned_bytes'] = 0
        _stage = stage
        stats['stage_switches'] += 1


def value_signature(value):
    # Native loaders recreate LowVramPatch wrappers on each stage switch.
    # Fingerprint their actual immutable patch contents, not wrapper identity.
    if isinstance(value, torch.Tensor):
        return ('tensor', id(value), value._version, value.shape, value.dtype, value.device)
    if type(value) is LoRAAdapter:
        return ('lora', value_signature(value.weights))
    if isinstance(value, (list, tuple)):
        return tuple(value_signature(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, value_signature(v)) for k, v in sorted(value.items()))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise RuntimeError(f'Unqualified retained LoRA patch value: {type(value)}')


def patch_signature(fn):
    if fn is None:
        return None
    if not retain_stages:
        return id(fn)
    if not isinstance(fn, comfy.model_patcher.LowVramPatch) or fn.convert_func is not None or fn.set_func is not None:
        raise RuntimeError('Unqualified retained LoRA patch wrapper')
    return (fn.key, value_signature(fn.patches[fn.key]))


def projection_signature(proj):
    return (value_signature(proj.weight), value_signature(proj.bias),
            patch_signature(getattr(proj, 'weight_lowvram_function', None)),
            patch_signature(getattr(proj, 'bias_lowvram_function', None)))


def install_prefetch_filter(blocks):
    # Native prefetch otherwise re-materializes full QKV weights even when the
    # owned effective copy already serves every QKV call. Filter only qualified
    # H3 QKV modules; preserve native prefetch and cleanup for all other weights.
    block_ids = {id(block) for block in blocks}
    native_make_queue = comfy.model_prefetch.make_prefetch_queue

    class FilteredBlock:
        def __init__(self, root):
            self.root = root

        def modules(self):
            proj = self.root.attn.qkv_proj
            signature = projection_signature(proj)
            cached = any(key[0] == _stage and key[1] == id(proj) and value[0] == signature
                         for key, value in _cache.items())
            for module in self.root.modules():
                if module is proj and cached:
                    stats['prefetch_skips'] += 1
                    continue
                yield module

    def make_queue(queue, device, transformer_options):
        filtered = [FilteredBlock(root) if id(root) in block_ids else root for root in queue]
        return native_make_queue(filtered, device, transformer_options)

    comfy.model_prefetch.make_prefetch_queue = make_queue


def effective_head_shard(proj, world, rank, example):
    if _stage is None:
        raise RuntimeError('Activate the native LoRA stage before QKV sharding')
    # These experiments qualify the native quantized dynamic-LoRA path only.
    if (proj.weight_function or proj.bias_function or proj._full_precision_mm
            or getattr(proj, 'comfy_force_cast_weights', False)
            or getattr(proj, 'layout_type', None) is None):
        raise RuntimeError('Unqualified effective QKV casting mode')
    signature = projection_signature(proj)
    key = (_stage, id(proj), world, rank, example.dtype, example.device)
    cached = _cache.get(key)
    if cached is not None and cached[0] == signature:
        stats['hits'] += 1
        return cached[1]
    if cached is not None:
        for i, field in enumerate(('weight', 'bias', 'weight_patch', 'bias_patch')):
            if cached[0][i] != signature[i]:
                reasons = stats['rebuild_reasons']
                reasons[field] = reasons.get(field, 0) + 1
    with comfy.ops.CastBiasWeightContext(proj, example, offloadable=True,
            compute_dtype=example.dtype, want_requant=True) as (weight, bias):
        if not isinstance(weight, QuantizedTensor):
            raise RuntimeError(f'Expected native requantized QKV, got {type(weight)}')
        layout = weight._layout_cls.__name__ if isinstance(weight._layout_cls, type) else weight._layout_cls
        if layout != 'TensorWiseINT8Layout' or getattr(weight._params, 'transposed', False):
            raise RuntimeError(f'Unqualified effective QKV layout {layout}')
        inner = weight.shape[0] // 3
        if inner % world:
            raise RuntimeError('QKV heads cannot be evenly sharded')
        span = inner // world
        idx = torch.cat([torch.arange(rank*span, (rank+1)*span, device=example.device)
                         + offset*inner for offset in range(3)])
        qdata = weight._qdata[idx].contiguous()
        scale = weight._params.scale
        if scale.numel() > 1:
            if scale.shape[0] != weight.shape[0]:
                raise RuntimeError('QKV scale is not per output row')
            scale = scale[idx].contiguous()
        else:
            scale = scale.clone()
        params = dataclasses.replace(weight._params, scale=scale,
                                     orig_shape=(qdata.shape[0], weight.shape[1]))
        owned = type(weight)(qdata, weight._layout_cls, params)
        owned_bias = bias[idx].contiguous() if bias is not None else None
        sub = copy.copy(proj)
        # A single-rank caller may wrap this instance's forward with the cache.
        # The owned copy must execute the native quantized Linear, not recurse.
        sub.forward = MethodType(type(proj).forward, sub)
        sub._parameters = dict(proj._parameters)
        sub._parameters.pop('weight', None)
        sub._parameters.pop('bias', None)
        sub._buffers = dict(proj._buffers)
        for name in ('_v', '_v_signature', '_prefetch', '_pin_state', '_sp_shard'):
            if hasattr(sub, name):
                delattr(sub, name)
        sub.weight_function = []
        sub.bias_function = []
        sub.weight_lowvram_function = None
        sub.bias_lowvram_function = None
        sub.weight = owned
        sub.bias = owned_bias
        sub.out_features = idx.numel()
        sub._orig_shape = (idx.numel(), weight.shape[1])
    owned_bytes = qdata.numel() * qdata.element_size() + scale.numel() * scale.element_size()
    if owned_bias is not None:
        owned_bytes += owned_bias.numel() * owned_bias.element_size()
    if cached is not None:
        stats['owned_bytes'] -= cached[2]
    _cache[key] = (signature, sub, owned_bytes)
    stats['owned_bytes'] += owned_bytes
    stats['builds'] += 1
    return sub
