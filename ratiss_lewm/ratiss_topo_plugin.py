"""External P_sig plugin for latent embeddings.

The plugin accepts torch tensors or NumPy arrays shaped (B,T,D), computes a
small Vietoris-Rips H1 persistence summary per sequence, and returns metrics
without importing or modifying LeWM. The preferred backend is ripser.
"""
from __future__ import annotations
import numpy as np

try:
    from ripser import ripser
except Exception:  # optional dependency
    ripser = None

def _as_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)

def _score(dgm):
    if dgm is None or len(dgm)==0:
        return 0.0
    finite = dgm[np.isfinite(dgm[:,1])]
    if len(finite)==0:
        return 0.0
    life = np.maximum(finite[:,1]-finite[:,0], 0.0)
    return float(life.max())

class TopologicalInvariantPlugin:
    """Post-encoder observable and optional differentiable loss adapter."""
    def __init__(self, max_points=96, maxdim=1, loss_weight=0.0):
        self.max_points=max_points; self.maxdim=maxdim; self.loss_weight=float(loss_weight)

    def measure(self, embeddings):
        z=_as_numpy(embeddings)
        if z.ndim==2: z=z[:,None,:]
        if z.ndim!=3: raise ValueError('embeddings must have shape (B,T,D) or (B,D)')
        values=[]; betti1=[]
        for seq in z:
            cloud=seq
            if len(cloud)>self.max_points:
                idx=np.linspace(0,len(cloud)-1,self.max_points).astype(int); cloud=cloud[idx]
            if ripser is None:
                values.append(0.0); betti1.append(0)
                continue
            dgms=ripser(cloud,maxdim=self.maxdim)['dgms']
            h1=dgms[1] if len(dgms)>1 else np.empty((0,2))
            finite=h1[np.isfinite(h1[:,1])] if len(h1) else h1
            values.append(_score(h1)); betti1.append(int(len(finite)))
        return {'p_sig': float(np.mean(values) if values else 0.0), 'p_sig_per_sequence': values, 'betti_h1': betti1, 'backend': 'ripser' if ripser else 'fallback'}

    def loss(self, embeddings):
        # Non-differentiable persistence is intentionally not inserted into the
        # core loss by default. This adapter returns a detached diagnostic term.
        result=self.measure(embeddings)
        return self.loss_weight * result['p_sig']
