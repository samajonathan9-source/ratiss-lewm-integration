
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

## 2026-09-07 — Cycle 3: calibration de sensibilité P_sig

- **Action:** niveaux d'interpolation 0,00 / 0,25 / 0,50 / 0,75 / 1,00 entre une boucle périodique et un nuage gaussien, graine 42.
- **Résultat:** P_sig max lifetime = 2,3569 / 0,8228 / 0,3859 / 0,3188 / 0,2972.
- **Mesure robuste:** le comptage H1 brut est conservé pour diagnostic ; la décision privilégie la durée de vie maximale et les barres dépassant 15 % du maximum.
- **Candidat d'inflexion:** environ 0,12 par plus forte baisse discrète ; ce candidat doit être validé sur des embeddings LeWM réels avant de devenir un seuil permanent.
- **Décision:** ne pas lancer l'A/B LeWM avant validation du seuil sur données latentes réelles.
- **Upstream changes:** none.
