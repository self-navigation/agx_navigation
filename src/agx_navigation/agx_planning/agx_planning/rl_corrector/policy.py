"""Load and run a trained RL corrector policy.

Keeps the torch and stable-baselines3 imports here so that a deployment
without a policy never imports torch: load_policy("") returns None and the
corrector stays identity. Prediction is deterministic; on any error or
non-finite action the caller falls back to identity (coefficient 1).
"""

from typing import Optional

import numpy as np


class Policy:
    """Thin wrapper over a trained SB3 SAC model for deterministic inference."""

    def __init__(self, model) -> None:
        self._model = model

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "Policy":
        # Imported here, not at module top, so torch only loads when a policy is
        # actually requested. Inference on-robot runs fine on CPU.
        from stable_baselines3 import SAC

        return cls(SAC.load(path, device=device))

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """obs -> action in [-1, 1]^action_dim (deterministic)."""
        action, _state = self._model.predict(np.asarray(obs, dtype=np.float32),
                                              deterministic=True)
        return np.asarray(action, dtype=np.float32)


def load_policy(path: str, device: str = "cpu") -> Optional[Policy]:
    """Load a policy, or return None if no path is set (-> identity fallback).

    Never raises: a missing/corrupt policy file logs nothing here and yields None,
    so the deployment seam stays identity rather than crashing the corrector.
    """
    if not path:
        return None
    try:
        return Policy.load(path, device=device)
    except Exception:
        return None
