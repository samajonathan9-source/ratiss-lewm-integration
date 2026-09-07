
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ratiss_lewm import TopologicalInvariantPlugin

rng=np.random.default_rng(42); t=np.linspace(0,2*np.pi,96,endpoint=False)
circle=np.c_[np.cos(t),np.sin(t)]; cloud=rng.normal(size=(96,2))
plugin=TopologicalInvariantPlugin(max_points=96)
rows=[]
for name,x in [('periodic_circle',circle),('noise',cloud)]:
    m=plugin.measure(x[None,:,:]); rows.append({'case':name,'p_sig':m['p_sig'],'betti_h1':m['betti_h1'][0],'backend':m['backend']})
Path('reports').mkdir(exist_ok=True)
with open('reports/ab_result_psig.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
Path('reports/test_report_psig.md').write_text('# Isolated P_sig test\n\nThe plugin was executed on a periodic 2D latent manifold and Gaussian noise. The report is an interface test; backend and numerical results are recorded in `ab_result_psig.csv`.\n',encoding='utf-8')
print(json.dumps(rows,indent=2))
