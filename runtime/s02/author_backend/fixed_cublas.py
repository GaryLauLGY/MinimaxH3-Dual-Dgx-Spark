"""Opt-in LMS route selection using the installed CUDA backend's own fallback.

No tensor arithmetic, quantizer, activation, allocator, or DLPack changes.
Installed only inside the isolated author child; not a production patch.
"""
import atexit
from collections import Counter
from contextvars import ContextVar
import functools


def install(sp, runner):
    import torch
    import comfy_kitchen.backends.cuda as cuda
    from comfy_kitchen.registry import registry

    inside_lms = ContextVar('h3_fixed_cublas_lms', default=False)
    selected = ContextVar('h3_fixed_cublas_selected', default=None)
    native_dispatch = registry.get_implementation
    native_preference = cuda._prefer_cublas_int8_fallback
    native_block = sp.sp_block
    roles = {(10752, 5376): 'qkv_shard', (5376, 7168): 'out_proj',
             (28672, 5376): 'mlp_fc1', (5376, 14336): 'mlp_fc2'}
    counts = Counter()
    lms_blocks = 0

    def report():
        value = dict(mode='fixed_native_cublas_fallback', lms_blocks=lms_blocks,
                     calls_by_role_mnk=dict(counts),
                     scope='All LMS blocks; four qualified BF16/no-bias/no-residual roles',
                     mathematical_changes=False, permanent_scratch_cache=False,
                     performance_claim=False)
        runner.record(f'fixed_cublas_rank{runner.a.rank}.json', value)
        return value

    def preference(m, n, k, device_index):
        shape = selected.get()
        if shape is not None and m > 1 and (n, k, device_index) == shape:
            counts[f'{roles[n, k]}:{m}:{n}:{k}'] += 1
            return True
        return native_preference(m, n, k, device_index)

    def dispatch(func_name, *args, **kwargs):
        native = native_dispatch(func_name, *args, **kwargs)
        if func_name != 'int8_linear' or not inside_lms.get() or native is not cuda.int8_linear:
            return native

        @functools.wraps(native)
        def linear(*values, **named):
            x, w = named.get('x'), named.get('weight')
            # All other signatures/dtypes/epilogues keep the native decision.
            eligible = (not values and isinstance(x, torch.Tensor)
                        and isinstance(w, torch.Tensor) and w.ndim == 2
                        and tuple(w.shape) in roles and x.is_cuda and w.is_cuda
                        and x.device == w.device and w.dtype == torch.int8
                        and x.dtype == torch.bfloat16
                        and named.get('out_dtype') == torch.bfloat16
                        and named.get('bias') is None and named.get('residual') is None)
            token = selected.set((*w.shape, x.get_device()) if eligible else None)
            try:
                return native(*values, **named)
            finally:
                selected.reset(token)
        return linear

    @functools.wraps(native_block)
    def block(*args, **kwargs):
        nonlocal lms_blocks
        is_lms = runner.stage_qkv._stage == 'lms'
        token = inside_lms.set(is_lms)
        try:
            result = native_block(*args, **kwargs)
            if is_lms:
                lms_blocks += 1
                # One small CPU report per 50-block forward. No GPU synchronization.
                if lms_blocks % 50 == 0:
                    report()
            return result
        finally:
            inside_lms.reset(token)

    cuda._prefer_cublas_int8_fallback = preference
    registry.get_implementation = dispatch
    sp.sp_block = block
    atexit.register(report)
    return report
