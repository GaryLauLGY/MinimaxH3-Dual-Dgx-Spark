"""Bounded candidate integration: both dispatch routes, activation and guards."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import comfy_kitchen as ck
import comfy_kitchen.backends.cuda as cuda
import fixed_cublas

ROOT = Path(__file__).resolve().parents[1]
torch.manual_seed(20261001)
torch.set_num_threads(8)
cases = []
for n, k, act in ((10752, 5376, None), (5376, 7168, None),
                  (28672, 5376, None), (5376, 14336, 'swiglu')):
    weight = torch.randint(-127, 128, (n, k), device='cuda', dtype=torch.int8)
    scale = torch.rand((n, 1), device='cuda')*.001 + .0001
    for m in (17, 19):
        x = torch.randn((m, k*(2 if act else 1)), device='cuda', dtype=torch.bfloat16)
        cases.append((x, weight, scale, act))

guard_x, guard_w, guard_scale, _ = cases[0]
guard_bias = torch.randn(guard_w.shape[0], device='cuda', dtype=torch.bfloat16)
guard_residual = torch.randn((guard_x.shape[0], guard_w.shape[0]), device='cuda', dtype=torch.bfloat16)
guard_residual_scale = torch.ones(guard_w.shape[0], device='cuda', dtype=torch.bfloat16)


def calculate(*args, **kwargs):
    results = []
    for x, w, scale, act in cases:
        if tuple(w.shape) == (28672, 5376):
            result = torch.ops.comfy_kitchen.int8_linear(x, w, scale, None, 2, True, 256)
        else:
            result = ck.int8_linear(x=x, weight=w, weight_scale=scale, out_dtype=torch.bfloat16,
                                   convrot=True, input_act=act)
        results.append(result)
    for extra in ({'bias': guard_bias},
                  {'residual': guard_residual, 'residual_scale': guard_residual_scale},
                  {'out_dtype': torch.float16}):
        options = dict(x=guard_x, weight=guard_w, weight_scale=guard_scale,
                       out_dtype=torch.bfloat16, convrot=True)
        options.update(extra)
        results.append(ck.int8_linear(**options))
    return results


oracle = calculate()
torch.cuda.synchronize()
records = {}


def record(name, value):
    records[name] = value
    (ROOT / ('selftest_'+name)).write_text(json.dumps(value, indent=2)+'\n')


sp = SimpleNamespace(sp_block=calculate)
runner = SimpleNamespace(a=SimpleNamespace(rank=0), stage_qkv=SimpleNamespace(_stage='turbo'), record=record)
flush = fixed_cublas.install(sp, runner)
cublas_calls = 0
original_cublas = cuda._C.cublas_gemm_int8


def counted(*args, **kwargs):
    global cublas_calls
    cublas_calls += 1
    return original_cublas(*args, **kwargs)


cuda._C.cublas_gemm_int8 = counted
turbo = sp.sp_block()
assert all(torch.equal(a, b) for a, b in zip(turbo, oracle))
assert cublas_calls == 0
runner.stage_qkv._stage = 'lms'
checks = []
for forward in (1, 2):
    got = sp.sp_block()
    torch.cuda.synchronize()
    checks.append([bool(torch.equal(a, b)) and bool(torch.isfinite(a).all()) for a, b in zip(got, oracle)])
assert all(all(row) for row in checks)
assert cublas_calls == 16, cublas_calls
# Direct calls outside an LMS block must retain the original CUTLASS route.
outside = calculate()
assert all(torch.equal(a, b) for a, b in zip(outside, oracle))
assert cublas_calls == 16
summary = flush()
assert sum(summary['calls_by_role_mnk'].values()) == 16
result = dict(passed=True, two_forwards_candidate_exact=checks,
              actual_cublas_calls=cublas_calls, turbo_and_outside_scope_unchanged=True,
              bias_residual_fp16_guards_exact_and_native=True,
              native_quantization_and_fc2_swiglu=True,
              source='Bounded synthetic integration; not full-model acceptance')
(ROOT/'selftest.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
