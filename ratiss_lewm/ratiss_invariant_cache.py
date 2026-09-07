"""Cache only latents that pass configured invariant checks."""
from __future__ import annotations
from collections import OrderedDict
class InvariantCache:
    def __init__(self,max_items=128, min_psig=0.0): self.max_items=max_items; self.min_psig=float(min_psig); self._data=OrderedDict()
    def put(self,key,latent,metrics):
        if float(metrics.get('p_sig',0.0))<self.min_psig: return False
        self._data[key]=(latent,dict(metrics)); self._data.move_to_end(key)
        while len(self._data)>self.max_items: self._data.popitem(last=False)
        return True
    def get(self,key):
        item=self._data.get(key)
        if item is not None: self._data.move_to_end(key)
        return item
    def __len__(self): return len(self._data)
