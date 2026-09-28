# Derived from buqi-code/buqi-minimax-h3-multigpu bce0839.
# Only attention/block primitives; current native H3 forward stays authoritative.
"""H3 sequence-parallel attention and transformer-block primitives.

Derived from buqi-code/buqi-minimax-h3-multigpu (MIT; see licenses/).
Only the heavy blocks are replaced. Embedding, final heads and audio scaling
remain in the installed native ComfyUI H3 forward. The public test pin is
recorded in docs/SETUP.md; do not infer compatibility from upstream versions.
"""

import copy
import dataclasses
import logging
import os
import time

import torch
import torch.distributed as dist

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.quant_ops
from comfy.ldm.minimax.model import (
    AUDIO_COND_TIMESTEP,
    VISUAL_COND_TIMESTEP,
    PackedLayout,
    pack_audio,
    patchify_video,
    rope_rotation_table,
    time_shift_sigma,
    unpack_audio,
    unpatchify_video,
)
from comfy.ldm.modules.attention import optimized_attention

EXPECTED_MODEL_SHA = "ee71d5c4993f29086b27fde1629a945ae48425bf"

PROFILE_OPS = bool(os.environ.get("MINIMAX_SP_PROFILE_OPS"))
AG_CHUNKS = max(1, int(os.environ.get("MINIMAX_SP_AG_CHUNKS", "4")))
ATTN_CHUNKS = max(1, int(os.environ.get("MINIMAX_SP_ATTN_CHUNKS", "4")))
_prof_acc = {}


class region:
    """Accumulate wall time per named region; only active under MINIMAX_SP_PROFILE_OPS."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if PROFILE_OPS:
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if PROFILE_OPS:
            torch.cuda.synchronize()
            _prof_acc[self.name] = _prof_acc.get(self.name, 0.0) + time.perf_counter() - self.t0
        return False


def prof_report(total):
    rows = sorted(_prof_acc.items(), key=lambda kv: -kv[1])
    body = "  ".join(f"{k}={v * 1000:.0f}ms" for k, v in rows)
    logging.info(f"[minimax_sp][ops] step={total * 1000:.0f}ms  {body}")
    _prof_acc.clear()


def use_allgather(world, hidden, inner):
    """Broadcasting h beats swapping Q/K/V while hidden < 3*inner/world.

    For H3 (hidden 5376, inner 7168) the two cost the same bytes at world 4 and
    below that the all_gather path additionally removes three full permute copies.
    """
    return 1 < world <= (3 * inner) // hidden


def head_sharded_qkv(proj, world, rank):
    """Row-slice qkv_proj down to this rank's heads, keeping fp8 storage.

    The weight carries one per-tensor scale, so selecting rows is exact: every
    output element remains the dot product it was on a single GPU. Cached on the
    module so it is freed with the model.
    """
    cached = getattr(proj, "_sp_shard", None)
    if cached is not None and cached[0] == (world, rank):
        return cached[1]

    w = proj.weight
    inner = w.shape[0] // 3
    span = inner // world
    lo = rank * span
    idx = torch.cat([torch.arange(lo, lo + span, device=w.device) + off * inner
                     for off in range(3)])

    sub = copy.copy(proj)
    sub._parameters = dict(proj._parameters)
    del sub._parameters["weight"]
    if type(w).__name__ == "QuantizedTensor":
        # INT8 ConvRot stores a separate scale for each output row. The
        # rotation acts along the input axis, so retain it and slice scales
        # together with output rows. Reject other block quantization layouts.
        layout = w._layout_cls.__name__ if isinstance(w._layout_cls, type) else str(w._layout_cls)
        if layout not in ("TensorWiseINT8Layout", "TensorCoreFP8E4M3Layout", "TensorCoreFP8E5M2Layout", "TensorCoreFP8Layout"):
            raise RuntimeError(f"Unqualified QKV sharding layout: {layout}")
        if getattr(w._params, "transposed", False):
            raise RuntimeError("Transposed quantized QKV storage is unsupported")
        qd = w._qdata[idx].contiguous()
        scale = w._params.scale
        if scale.numel() > 1:
            if scale.shape[0] != w.shape[0]:
                raise RuntimeError(f"Unexpected per-row QKV scale shape {scale.shape}")
            scale = scale[idx].contiguous()
        params = dataclasses.replace(w._params, scale=scale, orig_shape=(qd.shape[0], w.shape[1]))
        sub.weight = type(w)(qd, w._layout_cls, params)
    else:
        sub.weight = w[idx].contiguous()
    sub.out_features = idx.numel()
    if proj.bias is not None:
        sub._parameters.pop("bias", None)
        sub.bias = proj.bias[idx].contiguous()
    proj._sp_shard = ((world, rank), sub)
    return sub


def gather_rows_full(local, ctx):
    """[s_local, C] -> [S_total, C] in rank order, tolerating uneven row splits."""
    c = local.shape[1]
    maxn = max(ctx.splits)
    if min(ctx.splits) == maxn:
        out = torch.empty(ctx.world * maxn, c, dtype=local.dtype, device=local.device)
        dist.all_gather_into_tensor(out, local.contiguous(), group=ctx.group)
        return out
    buf = torch.zeros(maxn, c, dtype=local.dtype, device=local.device)
    buf[:local.shape[0]] = local
    out = torch.empty(ctx.world * maxn, c, dtype=local.dtype, device=local.device)
    dist.all_gather_into_tensor(out, buf, group=ctx.group)
    out = out.view(ctx.world, maxn, c)
    return torch.cat([out[r, :ctx.splits[r]] for r in range(ctx.world)], dim=0)


class SPContext:
    """Row-sharding descriptor for one forward pass."""

    def __init__(self, rank, world, group, seq_len):
        self.rank = rank
        self.world = world
        self.group = group
        self.seq_len = seq_len
        base, rem = divmod(seq_len, world)
        self.splits = [base + (1 if r < rem else 0) for r in range(world)]
        self.start = sum(self.splits[:rank])
        self.stop = self.start + self.splits[rank]

    @property
    def local(self):
        return self.splits[self.rank]


def post_heads_to_seq(t, ctx, pending):
    """[s_local, H, D] -> [S_total, H/P, D], transfer posted asynchronously.

    Appends (work, send) to pending: the send buffer has to stay referenced until
    the transfer completes. Posting rather than blocking lets the Q, K and V
    exchanges pipeline against each other's permute copies.
    """
    s_local, heads, dim = t.shape
    p = ctx.world
    hp = heads // p
    chunk = hp * dim
    send = t.reshape(s_local, p, chunk).transpose(0, 1).contiguous().view(-1)
    out = torch.empty(ctx.seq_len * chunk, dtype=t.dtype, device=t.device)
    work = dist.all_to_all_single(
        out, send,
        output_split_sizes=[n * chunk for n in ctx.splits],
        input_split_sizes=[s_local * chunk] * p,
        group=ctx.group,
        async_op=True,
    )
    pending.append((work, send))
    return out.view(ctx.seq_len, hp, dim)


def a2a_scatter_seq_gather_heads(t, ctx):
    """[S_total, (H/P)*D] -> [s_local, H*D]"""
    chunk = t.shape[1]
    p = ctx.world
    s_local = ctx.local
    out = torch.empty(p * s_local * chunk, dtype=t.dtype, device=t.device)
    dist.all_to_all_single(
        out, t.contiguous().view(-1),
        output_split_sizes=[s_local * chunk] * p,
        input_split_sizes=[n * chunk for n in ctx.splits],
        group=ctx.group,
    )
    return out.view(p, s_local, chunk).transpose(0, 1).reshape(s_local, p * chunk)


def gather_and_project(attn_proj, x, ctx, chunks, out_dim):
    """all_gather h and project it, with the transfers overlapping the projection.

    Chunk i holds the same row window from every rank, so its projected rows are
    written back at each rank's global offset and the result lands in exactly the
    row order a single GPU would produce. Row splits are uneven when the sequence
    does not divide by world, and all_gather needs equal contributions, so the
    local block is padded up to the largest split and the padding is dropped on
    write-back.
    """
    s_local, hidden = x.shape
    if chunks <= 1:
        return attn_proj(gather_rows_full(x, ctx))

    maxn = max(ctx.splits)
    if s_local == maxn:
        src = x
    else:
        src = torch.zeros(maxn, hidden, dtype=x.dtype, device=x.device)
        src[:s_local] = x

    span = -(-maxn // chunks)
    starts = [sum(ctx.splits[:r]) for r in range(ctx.world)]
    pending = []
    for a in range(0, maxn, span):
        b = min(a + span, maxn)
        buf = torch.empty(ctx.world * (b - a), hidden, dtype=x.dtype, device=x.device)
        src_chunk = src[a:b].contiguous()
        work = dist.all_gather_into_tensor(buf, src_chunk, group=ctx.group, async_op=True)
        pending.append((a, b, buf, work, src_chunk))

    out = torch.empty(ctx.seq_len, out_dim, dtype=x.dtype, device=x.device)
    for a, b, buf, work, _ in pending:
        work.wait()
        piece = attn_proj(buf)
        m = b - a
        for r in range(ctx.world):
            n = min(b, ctx.splits[r]) - a
            if n > 0:
                out[starts[r] + a:starts[r] + a + n] = piece[r * m:r * m + n]
    return out


def attention_and_exchange(q, k, v, ctx, transformer_options, chunks, hp, dim):
    """Attention in head chunks, with each chunk's output exchange in flight while
    the next chunk is computed.

    Heads are independent, so chunking the attention is exact. The exchange hands
    every rank the rows it owns for the chunk's heads; those land at that rank's
    head offset plus the chunk offset, which is the column order the single
    unchunked exchange produced.
    """
    inner = ctx.world * hp * dim
    if chunks <= 1:
        with region("attn"):
            out = optimized_attention(q, k, v, hp, mask=None, skip_reshape=True,
                                      transformer_options=transformer_options)
        with region("a2a_out"):
            return a2a_scatter_seq_gather_heads(out.squeeze(0), ctx)

    span = -(-hp // chunks)
    pending = []
    with region("attn+a2a_out"):
        for h0 in range(0, hp, span):
            h1 = min(h0 + span, hp)
            out = optimized_attention(q[:, h0:h1], k[:, h0:h1], v[:, h0:h1], h1 - h0,
                                      mask=None, skip_reshape=True,
                                      transformer_options=transformer_options)
            cd = (h1 - h0) * dim
            send = out.squeeze(0).contiguous().view(-1)
            buf = torch.empty(ctx.world * ctx.local * cd, dtype=q.dtype, device=q.device)
            work = dist.all_to_all_single(
                buf, send,
                output_split_sizes=[ctx.local * cd] * ctx.world,
                input_split_sizes=[n * cd for n in ctx.splits],
                group=ctx.group, async_op=True)
            pending.append((h0, cd, buf, work, send))

        out_full = torch.empty(ctx.local, inner, dtype=q.dtype, device=q.device)
        for h0, cd, buf, work, _ in pending:
            work.wait()
            piece = buf.view(ctx.world, ctx.local, cd)
            for r in range(ctx.world):
                off = r * hp * dim + h0 * dim
                out_full[:, off:off + cd] = piece[r]
    return out_full


def sp_attention_allgather(attn, x, rope_full, ctx, transformer_options, chunks=None):
    """Ulysses attention that broadcasts h instead of swapping Q/K/V.

    Each rank projects the whole sequence through only its own head rows, so Q/K/V
    come out already gathered over the sequence: no head/seq transpose, no QKV
    all-to-all, and only this rank's slice of qkv_proj has to be resident.
    """
    heads, dim = attn.heads, attn.head_dim
    hp = heads // ctx.world
    proj = head_sharded_qkv(attn.qkv_proj, ctx.world, ctx.rank)
    with region("gather+qkv_proj"):
        qkv = gather_and_project(proj, x, ctx, AG_CHUNKS if chunks is None else chunks,
                                 3 * hp * dim)
    s = qkv.shape[0]
    with region("qknorm+rope"):
        q, k, v = qkv.split(hp * dim, dim=-1)
        v = v.view(s, hp, dim)
        if rope_full is not None:
            q = q.view(1, s, hp, dim)
            k = k.view(1, s, hp, dim)
            qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
            rot = rope_full.shape[-3] * 2
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_full, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = attn.q_norm(q.view(s, hp, dim))
            k = attn.k_norm(k.view(s, hp, dim))
    out = attention_and_exchange(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0), ctx, transformer_options,
        ATTN_CHUNKS, hp, dim)
    with region("out_proj"):
        return attn.out_proj(out)


def sp_attention(attn, x, rope_freqs, ctx, transformer_options):
    s = x.shape[0]
    heads, dim = attn.heads, attn.head_dim
    with region("qkv_proj+qknorm+rope"):
        q, k, v = attn.qkv_proj(x).split(heads * dim, dim=-1)
        v = v.view(s, heads, dim)
        if rope_freqs is not None:
            q = q.view(1, s, heads, dim)
            k = k.view(1, s, heads, dim)
            qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = attn.q_norm(q.view(s, heads, dim))
            k = attn.k_norm(k.view(s, heads, dim))

    hp = heads // ctx.world
    with region("a2a_qkv"):
        pending = []
        q = post_heads_to_seq(q, ctx, pending)
        k = post_heads_to_seq(k, ctx, pending)
        v = post_heads_to_seq(v, ctx, pending)
        for work, _ in pending:
            work.wait()
        pending.clear()
    out = attention_and_exchange(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0), ctx, transformer_options,
        ATTN_CHUNKS, hp, dim)
    with region("out_proj"):
        return attn.out_proj(out)


def sp_block(block, x, t_emb, mod_segments, rope_freqs, ctx, transformer_options, allgather=False):
    from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift

    with region("adaln+norm+mod"):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
        h = _mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_segments)
    attn = sp_attention_allgather if allgather else sp_attention
    attn_out = attn(block.attn, h, rope_freqs, ctx, transformer_options)
    with region("adaln+norm+mod"):
        x = _mod_gate(x, gate_msa, attn_out, mod_segments)
        h = _mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, mod_segments)
    with region("mlp"):
        mlp_out = block.mlp(h)
    with region("adaln+norm+mod"):
        return _mod_gate(x, gate_mlp, mlp_out, mod_segments)


def shard_segments(segments, start, stop):
    """Clip global (start, stop, row) triples to a rank's row window, in local coords."""
    out = []
    for a, b, row in segments:
        lo, hi = max(a, start), min(b, stop)
        if hi > lo:
            out.append((lo - start, hi - start, row))
    return out
