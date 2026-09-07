"""Real-time latent cohesion monitor based on RATISS correlation observables."""
from __future__ import annotations
import numpy as np

class CohesionMonitor:
    def __init__(self, window=16): self.window=int(window); self.history=[]
    def update(self, embeddings):
        z=np.asarray(embeddings.detach().cpu() if hasattr(embeddings,'detach') else embeddings,dtype=float)
        if z.ndim==3: z=z.reshape(-1,z.shape[-1])
        if z.shape[0]<2: return {'entropy':0.0,'frustration':0.0,'regime_break':False}
        c=np.corrcoef(z,rowvar=False); c=np.nan_to_num(c); c=(c+c.T)/2; np.fill_diagonal(c,1.0)
        vals=np.abs(c[np.triu_indices_from(c,1)])
        hist=float(np.mean(vals)) if len(vals) else 0.0
        entropy=float(-np.mean(np.log(vals+1e-8))) if len(vals) else 0.0
        neg=float(np.mean(c[np.triu_indices_from(c,1)]<0)) if len(vals) else 0.0
        self.history.append(hist); self.history=self.history[-self.window:]
        baseline=float(np.mean(self.history[:-1])) if len(self.history)>1 else hist
        return {'mean_abs_correlation':hist,'entropy':entropy,'frustration':neg,'regime_break':bool(len(self.history)>3 and abs(hist-baseline)>3*np.std(self.history[:-1])+1e-9)}
