#!/usr/bin/env python3
"""Checkpoint-level A/B: real LeWM weights, deterministic synthetic pixel batch.

This is not a dataset/training benchmark: it validates the external RATISS
observability path on the actual public LeWM PushT checkpoint.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from transformers import ViTConfig, ViTModel

ROOT=Path(__file__).resolve().parents[1]
LEWM=Path('/home/ubuntu/lewm-integration-work/upstream/le-wm')
HF=Path('/home/ubuntu/lewm-integration-work/hf_pusht')
ENV_NOTE=str(LEWM)
sys.path.insert(0, str(LEWM))
sys.path.insert(0, str(ROOT))
from ratiss_lewm import TopologicalInvariantPlugin, ThermodynamicRegulator, CohesionMonitor

cfg=OmegaConf.load(HF/'config.json')
# Public checkpoint config targets installed stable_worldmodel classes.
model=instantiate(cfg)
# The public checkpoint stores a Hugging Face ViT encoder. The installed
# stable-pretraining release exposes a different internal encoder naming.
# Replace only the in-memory encoder with the compatible HF architecture.
vcfg=ViTConfig(hidden_size=192, num_hidden_layers=12, num_attention_heads=3, intermediate_size=768, image_size=224, patch_size=14, num_channels=3)
model.encoder=ViTModel(vcfg, add_pooling_layer=False)
state=torch.load(HF/'weights.pt',map_location='cpu',weights_only=False)
# Transformers 5 renamed ViT internals. Normalize the public checkpoint keys
# in memory only; no upstream source or checkpoint is modified.
key_map={
    '.encoder.layer.': '.layers.',
    '.attention.attention.query.': '.attention.q_proj.',
    '.attention.attention.key.': '.attention.k_proj.',
    '.attention.attention.value.': '.attention.v_proj.',
    '.attention.output.dense.': '.attention.o_proj.',
    '.intermediate.dense.': '.mlp.fc1.',
    '.output.dense.': '.mlp.fc2.',
}
normalized={}
for key,value in state.items():
    new_key=key
    for src,dst in key_map.items(): new_key=new_key.replace(src,dst)
    normalized[new_key]=value
state=normalized
model.load_state_dict(state,strict=True)
model.eval()
gen=torch.Generator(device='cpu').manual_seed(123)
B,T,C,H,W=2,4,3,224,224
pixels=torch.rand((B,T,C,H,W),generator=gen)
actions=torch.rand((B,T,10),generator=gen)
with torch.no_grad():
    encoded=model.encode({'pixels':pixels,'action':actions})
    emb=encoded['emb']
    pred=model.predict(emb[:,:3],encoded['act_emb'][:,:3])
    target=emb[:,1:4]
    vanilla_mse=float(torch.mean((pred-target)**2))

topo=TopologicalInvariantPlugin(max_points=64).measure(emb)
thermo=ThermodynamicRegulator().measure(pred,target)
cohesion=CohesionMonitor().update(emb)
result={'checkpoint':'quentinll/lewm-pusht','checkpoint_file_bytes':(HF/'weights.pt').stat().st_size,'input_shape':list(pixels.shape),'embedding_shape':list(emb.shape),'prediction_shape':list(pred.shape),'vanilla_next_embedding_mse':vanilla_mse,'ratiss_p_sig':topo['p_sig'],'ratiss_betti_h1':topo['betti_h1'],'ratiss_thermo':thermo,'ratiss_cohesion':cohesion,'note':'real LeWM checkpoint; deterministic synthetic pixels/actions; not full PushT dataset training'}
( ROOT/'reports/lewm_checkpoint_ab.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
with (ROOT/'reports/lewm_checkpoint_ab.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['vanilla_next_embedding_mse','ratiss_p_sig','ratiss_betti_h1','ratiss_energy','ratiss_entropy_proxy','ratiss_prediction_drift','ratiss_cohesion_entropy','ratiss_frustration'])
    w.writeheader(); w.writerow({'vanilla_next_embedding_mse':vanilla_mse,'ratiss_p_sig':topo['p_sig'],'ratiss_betti_h1':topo['betti_h1'],'ratiss_energy':thermo['energy'],'ratiss_entropy_proxy':thermo['entropy_proxy'],'ratiss_prediction_drift':thermo['prediction_drift'],'ratiss_cohesion_entropy':cohesion['entropy'],'ratiss_frustration':cohesion['frustration']})
print(json.dumps(result,indent=2))
