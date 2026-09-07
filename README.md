
# RATISS × LeWM external integration

This public repository contains external RATISS plugins and host-side adapters for experimenting with LeWM and stable-worldmodel. It deliberately does **not** modify the upstream repositories.

## Plugins

- `ratiss_topo_plugin.py`: P_sig-style persistent-homology observable for latent embeddings.
- `ratiss_eta_plugin.py`: thermodynamic-style energy/entropy regulator boundary.
- `ratiss_cohesion_plugin.py`: correlation entropy and frustration monitor.
- `ratiss_invariant_cache.py`: cache gate for validated latent states.
- `ratiss_hooks.py`: optional host-side adapter and PyTorch hook surface.

## Install

```bash
pip install -e .
# optional topological backend
pip install ripser
```

## Isolated test

```bash
python scripts/run_isolated_test.py
pytest -q
```

The first test is intentionally isolated from LeWM. It records backend availability and synthetic P_sig results in `reports/ab_result_psig.csv`.

## Integration rule

Use the plugins from an external training/evaluation harness. Do not edit `lucas-maes/le-wm` or `lucas-maes/stable-worldmodel`. Use `upstream_notes/ARCHITECTURE_AND_HOOKS.md` for the inspected boundaries and `INJECTION_LOG.md` for the experiment history.

## A/B protocol

Run a short vanilla-vs-plugin evaluation in a separate harness. Record convergence, prediction stability and topology-related metrics. If the plugin degrades the selected metrics, disable it and document the result in `INJECTION_LOG.md`.


## Current validation status

The isolated suite passes with `ripser` installed. The synthetic A/B scaffold can be run with:

```bash
python scripts/run_ab_protocol.py
```

This produces `reports/ab_result_synthetic.csv` and `.json`. It is not a claim of LeWM training improvement. A real LeWM A/B run must be performed in a compatible training environment using an external harness.

Source reuse and provenance are listed in `SOURCE_REUSE_MAP.md`.
