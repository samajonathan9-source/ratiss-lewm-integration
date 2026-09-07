#!/usr/bin/env python3
"""Deterministic sensitivity calibration for P_sig before any LeWM coupling."""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'; OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(42)
n=96
t=np.linspace(0,2*np.pi,n,endpoint=False)
circle=np.c_[np.cos(t),np.sin(t)]
noise=rng.normal(size=circle.shape)
levels=[0.0,0.25,0.50,0.75,1.0]
rows=[]
for level in levels:
    cloud=(1-level)*circle + level*noise
    cloud=(cloud-cloud.mean(axis=0))/(cloud.std(axis=0)+1e-12)
    dgms=ripser(cloud,maxdim=1)['dgms']
    h1=dgms[1]
    finite=h1[np.isfinite(h1[:,1])] if len(h1) else np.empty((0,2))
    life=np.maximum(finite[:,1]-finite[:,0],0.0) if len(finite) else np.array([])
    max_life=float(life.max()) if len(life) else 0.0
    # Robust count: ignore short bars below 15% of the longest observed bar.
    persistent_count=int(np.sum(life >= 0.15*max_life)) if max_life>0 else 0
    rows.append({'noise_level':level,'p_sig_max_lifetime':max_life,'raw_h1_count':int(len(finite)),'persistent_h1_count':persistent_count,'n_points':n})

# Discrete sensitivity: largest absolute drop between adjacent levels.
values=np.array([r['p_sig_max_lifetime'] for r in rows])
drops=np.diff(values)
idx=int(np.argmin(drops)) if len(drops) else 0
inflexion=float((levels[idx]+levels[idx+1])/2) if len(drops) else None
for r in rows: r['candidate_inflexion']=inflexion

with (OUT/'psig_sensitivity.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

fig,ax=plt.subplots(figsize=(7,4.2),dpi=160)
ax.plot(levels,values,'o-',color='#123b68',label='max H1 lifetime = P_sig')
ax.axvline(inflexion,color='#b23a48',linestyle='--',label=f'candidat inflexion ≈ {inflexion:.2f}')
ax.set_xlabel('Niveau de bruit / interpolation')
ax.set_ylabel('P_sig (durée H1 maximale)')
ax.set_title('Calibration de sensibilité P_sig avant couplage LeWM')
ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'psig_sensitivity.png'); plt.close(fig)

lines=['# Calibration de sensibilité P_sig','', '## Protocole', '', 'Nuage déterministe de 96 points : interpolation entre une boucle périodique et un nuage gaussien avec graine 42. Cinq niveaux sont mesurés : 0, 0,25, 0,50, 0,75 et 1,00. Le comptage brut H1 est conservé pour diagnostic, mais la décision utilise la durée maximale de vie H1 et un comptage robuste des barres dépassant 15 % de cette durée maximale.', '', '## Résultats', '', '| Bruit | P_sig max lifetime | H1 brut | H1 persistant robuste |', '|---:|---:|---:|---:|']
for r in rows: lines.append(f"| {r['noise_level']:.2f} | {r['p_sig_max_lifetime']:.6f} | {r['raw_h1_count']} | {r['persistent_h1_count']} |")
lines += ['', f'Le candidat d’inflexion discret est environ **{inflexion:.2f}**, défini par la plus forte baisse entre deux niveaux adjacents. Ce point est un seuil de calibration à confirmer sur des embeddings LeWM réels ; il ne doit pas encore être injecté comme seuil permanent.', '', '![Courbe de sensibilité](psig_sensitivity.png)', '', '## Décision', '', 'Ne pas lancer le test A/B LeWM avant inspection de la courbe et validation du seuil sur des embeddings réels. Le plugin doit utiliser la persistance des barres plutôt que le nombre H1 brut, qui est sensible aux petits cycles de bruit.']
(OUT/'psig_sensitivity_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(f'inflexion_candidate={inflexion:.2f}')
for r in rows: print(r)
