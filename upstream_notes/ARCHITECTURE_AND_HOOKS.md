
# LeWM architecture inspection (read-only)

The upstream repositories were cloned read-only. No upstream file is modified by this integration.

## Identified injection points

1. **Post-encoder observation:** `le-wm/jepa.py::JEPA.encode()` returns `info["emb"]`. An external wrapper can inspect this mapping after calling `encode`, or a caller can use a PyTorch forward hook on the encoder.
2. **Prediction boundary:** `le-wm/jepa.py::JEPA.predict()` returns predicted latent embeddings. A wrapper can compute thermodynamic/cohesion metrics after prediction without changing JEPA.
3. **Loss boundary:** `le-wm/train.py::lejepa_forward()` assembles `pred_loss`, `sigreg_loss`, and `loss`. A separate training harness can add a RATISS penalty to the returned loss, or log it as an auxiliary metric, without patching upstream.

## Stable-worldmodel boundary

`stable-worldmodel/stable_worldmodel/world.py::World` exposes environment reset, step, policy attachment and evaluation. The external integration should consume trajectories or latent batches produced by a host adapter rather than altering the environment package.

## Scope rule

This repository contains only external adapters, tests, reports and documentation. The upstream checkouts are not copied into the package and must remain untouched.
