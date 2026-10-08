"""Runtime re-join actor: a numpy forward pass of a td3_bc export.

Loads the .npz written by train_rejoin (MLP weights, state normalisation,
residual scale, n_knots). act() returns absolute (v, omega) knots over
linspace(0, T_w, n_knots), plan_knots + scale * tanh(MLP(s)), with s laid out
as rejoin_env.RejoinEnv.state. Integrating the knots is the caller's job.
"""

from __future__ import annotations

import numpy as np


class RejoinPolicy:
    def __init__(self, path: str):
        z = np.load(path)
        self.s_mean, self.s_std = z["s_mean"], z["s_std"]
        self.layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(int(z["n_layers"]))]
        self.scale = z["scale"]
        self.n_knots = int(z["n_knots"])

    def residual(self, s: np.ndarray) -> np.ndarray:
        x = (np.atleast_2d(s) - self.s_mean) / self.s_std
        for i, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        return np.tanh(x) * self.scale

    def act(self, s: np.ndarray) -> np.ndarray:
        s2 = np.atleast_2d(s)
        a = s2[:, 10:10 + 2 * self.n_knots] + self.residual(s2)
        return a if np.ndim(s) == 2 else a[0]
