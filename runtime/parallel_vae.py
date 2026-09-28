"""Parallel native H3 temporal VAE chunks; preserve native blending/output.

Based on the chunk-dispatch idea in buqi-code/buqi-minimax-h3-multigpu.
Uses the installed VAE's own temporal plan, and never modifies its weights.
"""
import torch
import torch.distributed as dist
import comfy.model_management as mm


def plan_and_prepare(fsm, z):
    padding, count = fsm._decode_temporal_chunks(z.shape[2])
    z = z * fsm.latents_std.view(1,-1,1,1,1).to(z) + fsm.latents_mean.view(1,-1,1,1,1).to(z)
    if padding:
        z = torch.cat([z,z[:,:,-1:].repeat(1,1,padding,1,1)],dim=2)
    bounds=[(i*fsm.tokens_chunk_size,min(i*fsm.tokens_chunk_size+fsm.tokens_chunk_size+fsm.token_overlap,z.shape[2])) for i in range(count)]
    return z,bounds


def decode_owned(fsm,z,bounds,rank):
    return {i:fsm._adaptive_decode(z[:,:,lo:hi]) for i,(lo,hi) in enumerate(bounds) if i%2==rank}


def load_gpu(vae):
    mm.load_models_gpu([vae.patcher],force_full_load=True)


def worker_decode(vae,meta,device):
    load_gpu(vae)
    z=torch.empty(meta['shape'],dtype=getattr(torch,meta['dtype']),device=device)
    dist.broadcast(z,src=0)
    prepared,bounds=plan_and_prepare(vae.first_stage_model,z)
    chunks=decode_owned(vae.first_stage_model,prepared,bounds,1)
    for i in sorted(chunks): dist.send(chunks[i].contiguous(),dst=0)
    return len(chunks)


def distributed_decode(vae,latent,obj_pg,device):
    fsm=vae.first_stage_model
    _,count=fsm._decode_temporal_chunks(latent.shape[2])
    if latent.shape[2]==1 or count<2:
        return vae.decode(latent)
    dist.broadcast_object_list([{'op':'vae_decode','shape':list(latent.shape),
                                'dtype':str(vae.vae_dtype).replace('torch.','')}],src=0,group=obj_pg)
    load_gpu(vae)
    z=latent.to(device=device,dtype=vae.vae_dtype).contiguous()
    dist.broadcast(z,src=0)
    prepared,bounds=plan_and_prepare(fsm,z)
    chunks=decode_owned(fsm,prepared,bounds,0)
    dtype=chunks[0].dtype
    for i,(lo,hi) in enumerate(bounds):
        if i%2:
            shape=(z.shape[0],3,(hi-lo)*fsm.vae_ratio_t,z.shape[3]*fsm.vae_ratio,z.shape[4]*fsm.vae_ratio)
            chunk=torch.empty(shape,dtype=dtype,device=device)
            dist.recv(chunk,src=1)
            chunks[i]=chunk
    original=fsm._adaptive_decode
    served=0
    def serve(clip):
        nonlocal served
        i=served;served+=1
        if i not in chunks or clip.shape[2]!=bounds[i][1]-bounds[i][0]:
            raise RuntimeError('Native VAE chunk order diverged from parallel plan')
        return chunks.pop(i)
    fsm._adaptive_decode=serve
    try:
        result=vae.decode(latent)
    finally:
        fsm._adaptive_decode=original
    if served!=len(bounds): raise RuntimeError('Native VAE did not consume all parallel chunks')
    return result
