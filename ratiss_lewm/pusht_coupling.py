"""RATISS sidecar coupling for LeWM on PushT.

This module keeps LeWM's vanilla prediction intact and adds an environment-aware
RATISS observation/control layer. It reuses the auditable RATISS topological
engine when available, while providing a deterministic fallback for Colab.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any
import numpy as np

from .ratiss_cohesion_plugin import CohesionMonitor
from .ratiss_eta_plugin import ThermodynamicRegulator
from .ratiss_invariant_cache import InvariantCache
from .ratiss_topo_plugin import TopologicalInvariantPlugin

# Optional source reuse from the user's RATISS topological engine repository.
_EXTERNAL = Path("/content/ratiss-topological-decoherence-engine")
if _EXTERNAL.exists():
    sys.path.insert(0, str(_EXTERNAL / "src"))
try:
    from ratiss_topological_decoherence.topology import topology_from_correlation
except Exception:  # pragma: no cover - fallback is tested locally
    topology_from_correlation = None


@dataclass(frozen=True)
class PushTLawConfig:
    """Deterministic transition-law parameters for dataset and rollout use."""

    # PushT actions are absolute 2-D target positions in the 0..512 workspace,
    # not velocity/delta commands.
    action_scale: float = 1.0
    workspace_scale: float = 512.0
    max_transition_error: float = 0.35
    min_coherence: float = 0.25
    psig_threshold: float = 0.12
    coupling_strength: float = 0.15
    cache_min_psig: float = 0.0


class PushTEnvironmentLaw:
    """Evaluate latent topology jointly with a physical PushT transition.

    The official PushT table exposes the agent's 2D state/action. The law uses
    that observed transition directly and never claims access to hidden block
    state. When a simulator supplies richer ``info`` fields, they are preserved
    in the returned record and can be used by a future environment-specific
    extension.
    """

    name = "pusht_transition_law_v1"

    def __init__(self, config: PushTLawConfig | None = None):
        self.config = config or PushTLawConfig()

    def topology(self, embedding_window: Any) -> dict[str, Any]:
        z = embedding_window.detach().cpu().numpy() if hasattr(embedding_window, "detach") else np.asarray(embedding_window)
        if z.ndim == 3:
            z = z[0]
        if z.ndim != 2 or z.shape[0] < 3:
            return {"p_sig": 0.0, "betti_h1": [0], "backend": "insufficient_window"}
        # The external engine's law is defined on correlation-profile geometry.
        corr = np.corrcoef(z, rowvar=True)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = (corr + corr.T) / 2.0
        np.fill_diagonal(corr, 1.0)
        if topology_from_correlation is not None:
            result = topology_from_correlation(corr)
            return {"p_sig": float(result["psig"]), "betti_h1": result["betti"], "backend": "ratiss-topological-decoherence-engine", "distance_model": result.get("distance_model")}
        # Fallback to the repository plugin when the source engine is absent.
        return TopologicalInvariantPlugin(max_points=96).measure(z[None, :, :]) | {"backend": "ratiss-lewm-plugin-fallback"}

    def oscillatory_coherence(self, embedding_window: Any, step: int = 0) -> float:
        """Use the RATISS LCT coherence law from the source repository.

        The source law oscillates *C* with ``cos(pi/2 * t)``. P_sig remains the
        topological persistence observable; it must not be silently conflated
        with C.
        """
        z = embedding_window.detach().cpu().numpy() if hasattr(embedding_window, "detach") else np.asarray(embedding_window)
        if z.ndim == 3:
            z = z[0]
        if z.ndim != 2 or len(z) == 0:
            return 0.0
        token = z[-1].reshape(-1)
        weights = z.mean(axis=0).reshape(-1)
        corr = float(np.dot(weights, token) / (np.linalg.norm(weights) * np.linalg.norm(token) + 1e-9))
        theta = math.cos((math.pi / 2.0) * int(step))
        return float(np.clip(abs(corr) * 5.0 * (0.5 + 0.5 * theta), 0.0, 1.0))

    def transition(self, state_t: Any, action_t: Any, state_tp1: Any, info: dict[str, Any] | None = None) -> dict[str, Any]:
        s0 = np.asarray(state_t, dtype=float).reshape(-1)
        a = np.asarray(action_t, dtype=float).reshape(-1)
        s1 = np.asarray(state_tp1, dtype=float).reshape(-1)
        n = min(len(s0), len(a), len(s1), 2)
        if n == 0:
            return {"physical_consistency": 0.0, "transition_error": float("inf"), "law": self.name}
        # In PushT, action[:2] is the absolute target position of the agent.
        # Comparing it to state delta made every transition look impossible.
        target = a[:n] * self.config.action_scale
        error_raw = float(np.linalg.norm(s1[:n] - target))
        error = error_raw / max(self.config.workspace_scale, 1e-9)
        coherence = float(np.exp(-error / max(self.config.max_transition_error, 1e-9)))
        record = {"physical_consistency": coherence, "transition_error": error, "transition_error_raw": error_raw, "target_norm": float(np.linalg.norm(target)), "state_delta_norm": float(np.linalg.norm(s1[:n] - s0[:n])), "law": self.name}
        if info:
            record["environment_info"] = dict(info)
        return record

    def evaluate(self, embedding_window: Any, predicted: Any, state_t: Any, action_t: Any, state_tp1: Any, reward: float | None = None, done: bool | None = None, info: dict[str, Any] | None = None) -> dict[str, Any]:
        topo = self.topology(embedding_window)
        transition = self.transition(state_t, action_t, state_tp1, info)
        step = int((info or {}).get("step", 0))
        transition["oscillatory_coherence"] = self.oscillatory_coherence(embedding_window, step)
        pred = predicted.detach().cpu().numpy() if hasattr(predicted, "detach") else np.asarray(predicted)
        transition["prediction_energy"] = float(np.mean(pred * pred))
        transition["reward"] = None if reward is None else float(reward)
        transition["done"] = None if done is None else bool(done)
        consistent = transition["physical_consistency"] >= self.config.min_coherence
        topologically_significant = float(topo.get("p_sig", 0.0)) >= self.config.psig_threshold
        gate = self.config.coupling_strength * transition["oscillatory_coherence"] if consistent else 0.0
        return {"topology": topo, "transition": transition, "consistent": bool(consistent), "topologically_significant": bool(topologically_significant), "coupling_gate": float(gate)}


class RATISSPushTCoupler:
    """LeWM-compatible sidecar: encode/predict remain vanilla, RATISS observes."""

    def __init__(self, law: PushTEnvironmentLaw | None = None):
        self.law = law or PushTEnvironmentLaw()
        self.topo = TopologicalInvariantPlugin(max_points=96)
        self.thermo = ThermodynamicRegulator()
        self.cohesion = CohesionMonitor()
        self.cache = InvariantCache(min_psig=self.law.config.cache_min_psig)

    def observe(self, encoded: dict[str, Any], predicted: Any, state_t: Any, action_t: Any, state_tp1: Any, reward: float | None = None, done: bool | None = None, info: dict[str, Any] | None = None) -> dict[str, Any]:
        emb = encoded["emb"]
        target = encoded.get("target_emb")
        metrics = self.law.evaluate(emb, predicted, state_t, action_t, state_tp1, reward, done, info)
        metrics["thermodynamics"] = self.thermo.measure(predicted, target)
        metrics["cohesion"] = self.cohesion.update(emb)
        accepted = self.cache.put(str(info.get("index", len(self.cache)) if info else len(self.cache)), emb.detach() if hasattr(emb, "detach") else emb, metrics["topology"])
        metrics["cache_accepted"] = bool(accepted)
        return metrics

    def auxiliary_weight(self, metrics: dict[str, Any]) -> float:
        """Return an environment gate; never replaces LeWM's vanilla MSE."""
        return float(metrics.get("coupling_gate", 0.0))
