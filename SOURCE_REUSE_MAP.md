# RATISS source reuse map

The external integration is based on existing RATISS interfaces and does not alter their repositories.

| Plugin | RATISS source consulted | Adaptation boundary |
|---|---|---|
| `ratiss_topo_plugin.py` | `ratiss_topo/robust_metrics.py`, `ratiss_topo/topology.py`, `ratiss-neuro/ratiss_neuro/topology.py` | latent batch `(B,T,D)` → P_sig/H1 metrics |
| `ratiss_eta_plugin.py` | `ratiss-bio/bio114/eth.py` thermodynamic-bath interface | latent energy/entropy telemetry and optional penalty |
| `ratiss_cohesion_plugin.py` | `ratiss_topo/robust_metrics.py`, `ratiss_topo/arsenal.py` | correlation entropy and signed-link frustration |
| `ratiss_invariant_cache.py` | RATISS validation/cache concept | admission gate for validated latent states |

The exact LeWM hook boundaries were identified from read-only upstream checkouts and are documented in `upstream_notes/ARCHITECTURE_AND_HOOKS.md`.
