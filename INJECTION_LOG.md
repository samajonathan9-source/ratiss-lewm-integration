
# RATISS × LeWM injection log

## 2026-09-07 — Cycle 0: read-only architecture inspection

- **Action:** Cloned `lucas-maes/le-wm` and `lucas-maes/stable-worldmodel` with shallow read-only checkouts.
- **Result:** Identified post-encoder, post-predictor and loss-assembly boundaries.
- **Change to upstream:** None.
- **Next:** Run the isolated P_sig plugin test.

## 2026-09-07 — Cycle 1: external plugin skeleton

- **Action:** Added `ratiss_topo_plugin.py`, `ratiss_eta_plugin.py`, `ratiss_cohesion_plugin.py`, `ratiss_invariant_cache.py` and `ratiss_hooks.py`.
- **Result:** External interfaces are importable without modifying LeWM.
- **Validation:** Unit tests and synthetic isolated test are prepared.
- **Next:** Execute isolated test, then design a host-side A/B harness.

## Experimental status

No claim of LeWM training improvement is made until the isolated test and a controlled A/B run are executed.

## 2026-09-07 — Cycle 2: isolated validation

- **Command:** `python scripts/run_isolated_test.py` and `pytest -q`.
- **Result:** 3 tests passed. With `ripser`, the periodic synthetic manifold produced `betti_h1=1`; the Gaussian control produced a different persistence profile.
- **A/B scaffold:** `scripts/run_ab_protocol.py` records vanilla MSE, P_sig, H1 count, drift and cohesion metrics on synthetic host-side predictions.
- **LeWM training:** not run yet; it requires the target environment and dataset dependencies.
- **Upstream changes:** none.
