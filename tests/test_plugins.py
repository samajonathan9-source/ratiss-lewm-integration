
import numpy as np
from ratiss_lewm import TopologicalInvariantPlugin, ThermodynamicRegulator, CohesionMonitor, InvariantCache

def test_topology_interface():
    t=np.linspace(0,2*np.pi,64,endpoint=False)
    circle=np.c_[np.cos(t),np.sin(t)]
    out=TopologicalInvariantPlugin(max_points=64).measure(circle[None,:,:])
    assert 'p_sig' in out and 'betti_h1' in out

def test_thermo_and_cohesion():
    x=np.random.default_rng(0).normal(size=(2,16,8))
    assert ThermodynamicRegulator().measure(x)['energy'] >= 0
    assert 'entropy' in CohesionMonitor().update(x)

def test_cache_gate():
    c=InvariantCache(min_psig=0.5)
    assert not c.put('bad',1,{'p_sig':0.1})
    assert c.put('good',2,{'p_sig':0.9})
    assert c.get('good')[0]==2
