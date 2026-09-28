# Derived tensor protocol only; no local worker or service lifecycle.
"""Tensor metadata protocol derived from buqi-code's MIT implementation.

Both ranks are isolated processes, one CUDA device per host. Rank 0 broadcasts
metadata through Gloo and tensors through NCCL. No production ComfyUI server
is patched. Rank 1 participates in the DiT and optionally native VAE chunks.
"""

import atexit
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import comfy.model_management

import sp_primitives as spf


REF_META_KEYS = ("kind", "latent_h", "latent_w", "latent_t", "ref_audio_t")
TO_WORKER_OPTIONS = ("minimax_h3_sigma_shift_video", "minimax_h3_sigma_shift_audio")

_GROUPS = {}


def tensor_meta(t):
    if t is None:
        return None
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}


def alloc_from_meta(meta, device):
    if meta is None:
        return None
    return torch.empty(meta["shape"], dtype=getattr(torch, meta["dtype"]), device=device)


def build_meta(x, timestep, context, transformer_options, payload):
    payload = payload or {}
    cond_v = payload.get("cond_video_latents", []) or []
    cond_a = payload.get("cond_audio_latents", []) or []
    tags = payload.get("text_token_tags")
    return {
        "op": "forward",
        "video": tensor_meta(x[0]),
        "audio": tensor_meta(x[1]),
        "timestep": tensor_meta(timestep),
        "context": tensor_meta(context),
        "tags": tensor_meta(tags),
        "cond_video": [tensor_meta(t) for t in cond_v],
        "cond_audio": [tensor_meta(t) for t in cond_a],
        "keyframes": [{"resolved_frame_index": kf["resolved_frame_index"]}
                      for kf in (payload.get("keyframes") or [])] or None,
        "refs": [{k: r[k] for k in REF_META_KEYS if k in r}
                 for r in (payload.get("refs") or [])] or None,
        "frame_count": payload.get("frame_count"),
        "seed": payload.get("seed", 0),
        "visual_cond_noise_aug": payload.get("visual_cond_noise_aug", spf.VISUAL_COND_TIMESTEP),
        "audio_cond_noise_aug": payload.get("audio_cond_noise_aug", spf.AUDIO_COND_TIMESTEP),
        "options": {k: transformer_options[k] for k in TO_WORKER_OPTIONS if k in transformer_options},
    }


def payload_from_meta(meta, tensors):
    p = {
        "seed": meta["seed"],
        "visual_cond_noise_aug": meta["visual_cond_noise_aug"],
        "audio_cond_noise_aug": meta["audio_cond_noise_aug"],
    }
    if meta["keyframes"]:
        p["keyframes"] = meta["keyframes"]
    if meta["refs"]:
        p["refs"] = meta["refs"]
    if meta["frame_count"] is not None:
        p["frame_count"] = meta["frame_count"]
    if tensors["tags"] is not None:
        p["text_token_tags"] = tensors["tags"]
    if tensors["cond_video"]:
        p["cond_video_latents"] = tensors["cond_video"]
    if tensors["cond_audio"]:
        p["cond_audio_latents"] = tensors["cond_audio"]
    return p


def ordered_tensors(x, timestep, context, payload):
    payload = payload or {}
    out = [x[0], x[1], timestep, context, payload.get("text_token_tags")]
    out += list(payload.get("cond_video_latents", []) or [])
    out += list(payload.get("cond_audio_latents", []) or [])
    return out
