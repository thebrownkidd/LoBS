"""PPO with a hybrid action: a Bernoulli "throw now?" head and a 3-D Gaussian for the throw parameters (A11).

The Gaussian only matters on the step where the policy throws, so its log-probability and entropy are weighted by
the throw indicator / throw probability."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import constants as C

LOG_2PI = math.log(2.0 * math.pi)


def _mlp(inp: int, hidden: int, out: int, out_gain: float) -> nn.Sequential:
    net = nn.Sequential(
        nn.Linear(inp, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out),
    )
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, math.sqrt(2.0))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(net[-1].weight, out_gain)
    return net


class Policy(nn.Module):
    def __init__(self, obs_dim: int = C.OBS_DIM, hidden: int = C.HIDDEN):
        super().__init__()
        self.actor = _mlp(obs_dim, hidden, 4, 0.01)  # [throw logit, 3 means]
        self.critic = _mlp(obs_dim, hidden, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((3,), C.LOG_STD_INIT))
        with torch.no_grad():
            self.actor[-1].bias[0] = C.THROW_BIAS_INIT

    # ------------------------------------------------------------------ distribution pieces
    def _heads(self, obs):
        out = self.actor(obs)
        return out[..., 0], out[..., 1:]

    def _logp(self, logit, mean, throw, praw):
        lp_throw = -nn.functional.binary_cross_entropy_with_logits(logit, throw, reduction="none")
        std = self.log_std.exp()
        lp_gauss = (-0.5 * ((praw - mean) / std) ** 2 - self.log_std - 0.5 * LOG_2PI).sum(-1)
        return lp_throw + throw * lp_gauss

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        """Sample (or take the mode of) an action. Returns env actions (N,4), throw (N,), raw params (N,3),
        log-prob (N,), value (N,)."""
        logit, mean = self._heads(obs)
        if deterministic:
            throw = (logit >= 0.0).float()
            praw = mean
        else:
            throw = torch.bernoulli(torch.sigmoid(logit))
            praw = mean + self.log_std.exp() * torch.randn_like(mean)
        env_a = torch.cat([(throw * 2.0 - 1.0)[:, None], praw.clamp(-1.0, 1.0)], dim=1)
        return env_a, throw, praw, self._logp(logit, mean, throw, praw), self.value(obs)

    def evaluate(self, obs, throw, praw):
        """Log-prob, entropy and value of stored actions (gradients flow)."""
        logit, mean = self._heads(obs)
        p = torch.sigmoid(logit)
        h_throw = nn.functional.binary_cross_entropy_with_logits(logit, p, reduction="none")
        h_gauss = (0.5 + 0.5 * LOG_2PI + self.log_std).sum()
        return self._logp(logit, mean, throw, praw), h_throw + p * h_gauss, self.value(obs)

    def p_throw(self, obs):
        with torch.no_grad():
            return torch.sigmoid(self._heads(obs)[0])


@dataclass
class Rollout:
    obs: torch.Tensor
    throw: torch.Tensor
    praw: torch.Tensor
    logp: torch.Tensor
    adv: torch.Tensor
    ret: torch.Tensor
    val: torch.Tensor


def gae(rew, val, done, last_val, gamma: float = C.GAMMA, lam: float = C.GAE_LAMBDA):
    """rew, val, done: (T,N). ``done[t]`` marks that the step at t ended its episode. Returns advantages, returns."""
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    last = torch.zeros_like(last_val)
    for t in reversed(range(T)):
        nxt_val = last_val if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nxt_val * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def update(policy: Policy, opt: torch.optim.Optimizer, ro: Rollout, epochs: int = C.EPOCHS,
           minibatches: int = C.MINIBATCHES, clip: float = C.CLIP, ent_coef: float = C.ENT_COEF,
           vf_coef: float = C.VF_COEF, max_grad_norm: float = C.MAX_GRAD_NORM,
           target_kl: float = C.TARGET_KL) -> dict:  # fmt: skip
    n = ro.obs.shape[0]
    mb = max(n // minibatches, 1)
    stats = {"kl": 0.0, "clipfrac": 0.0, "pg": 0.0, "vf": 0.0, "ent": 0.0, "epochs": 0}
    stop = False
    for ep in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n - mb + 1, mb):
            idx = perm[s : s + mb]
            lp, ent, v = policy.evaluate(ro.obs[idx], ro.throw[idx], ro.praw[idx])
            log_ratio = lp - ro.logp[idx]
            ratio = log_ratio.exp()
            a = ro.adv[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            pg = torch.max(-a * ratio, -a * ratio.clamp(1 - clip, 1 + clip)).mean()
            vf = 0.5 * ((v - ro.ret[idx]) ** 2).mean()
            loss = pg + vf_coef * vf - ent_coef * ent.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            opt.step()
            with torch.no_grad():
                kl = ((ratio - 1) - log_ratio).mean().item()
            stats["kl"], stats["pg"], stats["vf"], stats["ent"] = (
                kl,
                pg.item(),
                vf.item(),
                ent.mean().item(),
            )
            stats["clipfrac"] = ((ratio - 1).abs() > clip).float().mean().item()
            if kl > 1.5 * target_kl:
                stop = True
                break
        stats["epochs"] = ep + 1
        if stop:
            break
    return stats


def to_tensor(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)
