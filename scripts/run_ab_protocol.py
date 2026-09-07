"""Lightweight host-side A/B protocol; does not import or alter LeWM."""
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ratiss_lewm import TopologicalInvariantPlugin, CohesionMonitor, ThermodynamicRegulator

rng=np.random.default_rng(7)
rows=[]
for seed in range(4):
    r=np.random.default_rng(seed); t=np.linspace(0,2*np.pi,64,endpoint=False)
    base=np.c_[np.cos(t),np.sin(t)] + 0.05*r.normal(size=(64,2))
    pred=base + 0.08*r.normal(size=base.shape)
    vanilla=float(np.mean((pred-base)**2))
    topo=TopologicalInvariantPlugin(max_points=64).measure(pred[None,:,:])
    thermo=ThermodynamicRegulator().measure(pred,base)
    coh=CohesionMonitor().update(pred)
    rows.append({'seed':seed,'vanilla_mse':vanilla,'plugin_psig':topo['p_sig'],'betti_h1':topo['betti_h1'][0],'prediction_drift':thermo['prediction_drift'],'cohesion_entropy':coh['entropy']})
Path('reports').mkdir(exist_ok=True)
with open('reports/ab_result_synthetic.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
Path('reports/ab_result_synthetic.json').write_text(json.dumps({'protocol':'synthetic host-side A/B scaffold','rows':rows,'warning':'not a LeWM training result'},indent=2))
print(json.dumps(rows,indent=2))
