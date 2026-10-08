"""TD3, eps-TD3+BC, TD3+BC and BC learners for the one-step re-join, in torch.

The actor maximises lam * Q(s, pi(s)) - beta(s) * m(s) * ||pi(s) - a_PMP||^2
with beta(s) = beta0 * exp(-eps_hat(s) / eps0), where eps_hat is how much more
the PMP teacher's action costs under the plant's chi than under nominal chi,
and m(s) masks failed solves. Variant.kind picks td3 (beta0 = 0), td3bc
(constant beta), eps_td3bc or bc (no critic). Actions are in normalised
[-1, 1] units. Since gamma = 0 there are no target networks; twin critics with
a Huber loss and delayed actor updates are kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KINDS = ("td3", "td3bc", "eps_td3bc", "bc")


@dataclass
class Variant:
    kind: str = "eps_td3bc"
    alpha: float = 2.5          # TD3+BC's lam = alpha / mean|Q|
    beta0: float = 1.0
    eps0: float = float("nan")  # set from data (median teacher gap) when NaN
    hidden: int = 256
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    policy_delay: int = 2
    huber_delta: float = 5.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")


def mlp(i, o, h):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    """Normalises the state itself (mean/sd are buffers), so the exported
    weights are the whole policy."""

    def __init__(self, sd, ad, h, s_mean, s_std):
        super().__init__()
        self.net = mlp(sd, ad, h)
        self.register_buffer("s_mean", torch.as_tensor(s_mean, dtype=torch.float32))
        self.register_buffer("s_std", torch.as_tensor(s_std, dtype=torch.float32))
        # start at the zero residual (= follow the plan's twist from x0)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s):
        return torch.tanh(self.net((s - self.s_mean) / self.s_std))


class Critic(nn.Module):
    def __init__(self, sd, ad, h, s_mean, s_std):
        super().__init__()
        self.q1, self.q2 = mlp(sd + ad, 1, h), mlp(sd + ad, 1, h)
        self.register_buffer("s_mean", torch.as_tensor(s_mean, dtype=torch.float32))
        self.register_buffer("s_std", torch.as_tensor(s_std, dtype=torch.float32))

    def forward(self, s, a):
        x = torch.cat([(s - self.s_mean) / self.s_std, a], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def beta_of(variant: Variant, eps_hat: torch.Tensor) -> torch.Tensor:
    if variant.kind == "td3":
        return torch.zeros_like(eps_hat)
    if variant.kind in ("td3bc", "bc"):
        return torch.full_like(eps_hat, variant.beta0)
    return variant.beta0 * torch.exp(-eps_hat.clamp(min=0.0) / variant.eps0)


class Learner:
    def __init__(self, sd, ad, variant: Variant, s_mean, s_std, device="cpu"):
        if variant.kind == "eps_td3bc" and not math.isfinite(variant.eps0):
            raise ValueError("eps_td3bc needs eps0")
        self.v = variant
        self.dev = torch.device(device)
        self.actor = Actor(sd, ad, variant.hidden, s_mean, s_std).to(self.dev)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=variant.lr_actor)
        self.critic = None
        if variant.kind != "bc":
            self.critic = Critic(sd, ad, variant.hidden, s_mean, s_std).to(self.dev)
            self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=variant.lr_critic)
        self.n_updates = 0

    @torch.no_grad()
    def act(self, s: np.ndarray) -> np.ndarray:
        return self.actor(torch.as_tensor(s, dtype=torch.float32, device=self.dev)).cpu().numpy()

    def update(self, batch: dict) -> dict:
        """One gradient step. ``batch``: s, a (normalised), r (= -cost),
        a_bc (normalised teacher residual), m (label mask), eps_hat."""
        t = {k: torch.as_tensor(v, dtype=torch.float32, device=self.dev) for k, v in batch.items()}
        info = {}
        if self.critic is not None:
            q1, q2 = self.critic(t["s"], t["a"])
            lc = (F.huber_loss(q1, t["r"], delta=self.v.huber_delta)
                  + F.huber_loss(q2, t["r"], delta=self.v.huber_delta))
            self.opt_c.zero_grad()
            lc.backward()
            self.opt_c.step()
            info["critic_loss"] = lc.item()
        self.n_updates += 1
        if self.critic is not None and self.n_updates % self.v.policy_delay:
            return info
        pi = self.actor(t["s"])
        bc_err = ((pi - t["a_bc"]) ** 2).mean(-1)
        beta = beta_of(self.v, t["eps_hat"]) * t["m"]
        l_bc = (beta * bc_err).mean()
        if self.critic is None:
            la = l_bc
        else:
            q = self.critic(t["s"], pi)[0]
            lam = self.v.alpha / q.abs().mean().detach().clamp(min=1e-6)
            la = -lam * q.mean() + l_bc
            info["q_pi"] = q.mean().item()
            info["lam"] = lam.item()
        self.opt_a.zero_grad()
        la.backward()
        self.opt_a.step()
        info.update(actor_loss=la.item(), bc_err=bc_err.mean().item(), beta_mean=beta.mean().item())
        return info

    def export(self) -> dict:
        """numpy weights for rejoin_policy (no torch at runtime)."""
        sd = {k: v.detach().cpu().numpy() for k, v in self.actor.state_dict().items()}
        lin = [m for m in self.actor.net if isinstance(m, nn.Linear)]
        out = dict(s_mean=sd["s_mean"], s_std=sd["s_std"], n_layers=np.array(len(lin)))
        for i, m in enumerate(lin):
            out[f"W{i}"] = m.weight.detach().cpu().numpy()
            out[f"b{i}"] = m.bias.detach().cpu().numpy()
        return out
