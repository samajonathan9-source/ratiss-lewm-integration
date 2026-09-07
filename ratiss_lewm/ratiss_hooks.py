"""Non-invasive hook adapters for LeWM-like objects."""
from __future__ import annotations
from dataclasses import dataclass, field
from .ratiss_topo_plugin import TopologicalInvariantPlugin
from .ratiss_eta_plugin import ThermodynamicRegulator
from .ratiss_cohesion_plugin import CohesionMonitor

@dataclass
class RATISSHookBundle:
    topo: TopologicalInvariantPlugin = field(default_factory=TopologicalInvariantPlugin)
    thermo: ThermodynamicRegulator = field(default_factory=ThermodynamicRegulator)
    cohesion: CohesionMonitor = field(default_factory=CohesionMonitor)
    def post_encode(self, output):
        emb=output['emb'] if isinstance(output,dict) else output
        metrics={'topology':self.topo.measure(emb),'cohesion':self.cohesion.update(emb)}
        output['ratiss']=metrics if isinstance(output,dict) else metrics
        return output
    def post_predict(self, predicted, target=None):
        return {'thermodynamics':self.thermo.measure(predicted,target),'penalty':self.thermo.penalty(predicted,target)}

def attach_hooks(model, bundle=None):
    """Attach observational forward hooks when the host exposes an encoder.

    This function is opt-in and returns handles; it never patches source files.
    """
    bundle=bundle or RATISSHookBundle(); handles=[]
    encoder=getattr(model,'encoder',None)
    if encoder is not None and hasattr(encoder,'register_forward_hook'):
        def _hook(_module,_inputs,output):
            return output
        handles.append(encoder.register_forward_hook(_hook))
    return bundle, handles
