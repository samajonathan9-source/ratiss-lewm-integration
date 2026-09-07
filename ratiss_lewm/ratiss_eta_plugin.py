"""External thermodynamic-style latent regulator.

This is an adapter boundary: it measures normalized energy and entropy of
latent predictions and exposes a penalty that a caller may add to its own loss.
It does not alter LeWM internals.
"""
from __future__ import annotations
import numpy as np

class ThermodynamicRegulator:
    def __init__(self, target_energy=None, entropy_floor=0.0, weight=0.0):
        self.target_energy=target_energy; self.entropy_floor=float(entropy_floor); self.weight=float(weight)
    def measure(self, predicted, target=None):
        p=np.asarray(predicted.detach().cpu() if hasattr(predicted,'detach') else predicted, dtype=float)
        energy=float(np.mean(p*p))
        centered=p-p.mean(axis=-1, keepdims=True)
        variance=float(np.mean(centered*centered))
        entropy=float(0.5*np.log(2*np.pi*np.e*max(variance,1e-12)))
        drift=None
        if target is not None:
            t=np.asarray(target.detach().cpu() if hasattr(target,'detach') else target, dtype=float)
            drift=float(np.mean((p-t)**2))
        return {'energy':energy,'entropy_proxy':entropy,'prediction_drift':drift}
    def penalty(self, predicted, target=None):
        m=self.measure(predicted,target)
        e=0.0 if self.target_energy is None else (m['energy']-self.target_energy)**2
        s=max(self.entropy_floor-m['entropy_proxy'],0.0)**2
        return float(self.weight*(e+s))
