import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.distributions import Bernoulli, Normal  # noqa: E402

from demos.lobpass import constants as C  # noqa: E402
from demos.lobpass import curriculum as cur  # noqa: E402
from demos.lobpass import ppo  # noqa: E402
from demos.lobpass.env import LobPassEnv  # noqa: E402
from demos.lobpass.train import load_policy, save_ckpt  # noqa: E402


def test_hybrid_logprob_and_entropy_match_brute_force():
    torch.manual_seed(0)
    pol = ppo.Policy()
    with torch.no_grad():
        pol.actor[-1].weight.mul_(30.0)  # make the heads non-trivial
        pol.log_std.copy_(torch.tensor([-0.5, -1.0, 0.2]))
    obs = torch.randn(32, C.OBS_DIM)
    throw = torch.bernoulli(torch.full((32,), 0.5))
    praw = torch.randn(32, 3)
    lp, ent, _ = pol.evaluate(obs, throw, praw)
    logit, mean = pol._heads(obs)
    ref = Bernoulli(logits=logit).log_prob(throw) + throw * Normal(
        mean, pol.log_std.exp()
    ).log_prob(praw).sum(-1)
    assert torch.allclose(lp, ref, atol=1e-5)
    p = torch.sigmoid(logit)
    ref_ent = Bernoulli(logits=logit).entropy() + p * Normal(mean, pol.log_std.exp()).entropy().sum(
        -1
    )
    assert torch.allclose(ent, ref_ent, atol=1e-5)


def test_act_is_consistent_with_evaluate_and_deterministic_mode():
    torch.manual_seed(1)
    pol = ppo.Policy()
    obs = torch.randn(16, C.OBS_DIM)
    a, throw, praw, lp, _ = pol.act(obs)
    lp2, _, _ = pol.evaluate(obs, throw, praw)
    assert torch.allclose(lp, lp2, atol=1e-5)
    assert set(a[:, 0].tolist()) <= {-1.0, 1.0} and a[:, 1:].abs().max() <= 1.0
    d1, *_ = pol.act(obs, deterministic=True)
    d2, *_ = pol.act(obs, deterministic=True)
    assert torch.equal(d1, d2)
    assert float(pol.p_throw(obs).mean()) < 0.2  # the -2 bias: rare throws at the start


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(2)
    pol = ppo.Policy()
    opt = torch.optim.Adam(pol.parameters())
    save_ckpt(tmp_path / "c.pt", pol, opt, "S0", 123, 123, 0, {"ppo": {"hidden": C.HIDDEN}})
    back = load_policy(tmp_path / "c.pt")
    obs = torch.randn(4, C.OBS_DIM)
    assert torch.equal(pol.actor(obs), back.actor(obs)) and torch.equal(pol.log_std, back.log_std)


def test_gate_promotes_only_on_two_consecutive_passes():
    g = cur.Gate("S0", None)
    assert g.threshold == C.GATES["S0"]
    assert [g.update(x) for x in (0.99, 0.5, 0.99, 0.99)] == [False, False, False, True]
    g2 = cur.Gate("S2", 0.8)
    assert g2.threshold == pytest.approx(0.85 * 0.8)


def test_stage_mix_proportions():
    assert cur.stage_mix("S0") == {"S0": 1.0}
    m = cur.stage_mix("S3b")
    assert (
        m["S3b"] == 0.75 and sum(m.values()) == pytest.approx(1.0) and set(m) == set(C.STAGES[:5])
    )


def test_gae_matches_hand_computation():
    rew = torch.tensor([[0.0], [0.0], [1.0]])
    val = torch.tensor([[0.5], [0.4], [0.3]])
    done = torch.tensor([[0.0], [0.0], [1.0]])
    adv, ret = ppo.gae(rew, val, done, torch.tensor([0.0]), gamma=0.9, lam=1.0)
    # lam = 1: advantage = discounted return - value
    assert adv[2].item() == pytest.approx(1.0 - 0.3)
    assert adv[1].item() == pytest.approx(0.9 * 1.0 - 0.4)
    assert adv[0].item() == pytest.approx(0.81 * 1.0 - 0.5)
    assert torch.allclose(ret, adv + val)


def test_two_env_smoke_two_updates_change_the_policy():
    torch.manual_seed(0)
    envs = [LobPassEnv({"S0": 1.0}, base_seed=0, rank=r) for r in range(2)]
    pol = ppo.Policy()
    opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    before = [p.detach().clone() for p in pol.parameters()]
    obs = torch.as_tensor(np.stack([e.reset()[0] for e in envs]))
    for _ in range(2):
        T = 48
        ob, th, pr, lp, va, rw, dn = [], [], [], [], [], [], []
        for _t in range(T):
            a, throw, praw, logp, val = pol.act(obs)
            nxt, r, d = [], [], []
            for i, e in enumerate(envs):
                o, rr, term, _, _ = e.step(a[i].numpy())
                nxt.append(e.reset()[0] if term else o)
                r.append(rr)
                d.append(float(term))
            ob.append(obs), th.append(throw), pr.append(praw), lp.append(logp), va.append(val)
            rw.append(torch.tensor(r, dtype=torch.float32)), dn.append(torch.tensor(d))
            obs = torch.as_tensor(np.stack(nxt))
        rw, va, dn = torch.stack(rw), torch.stack(va), torch.stack(dn)
        adv, ret = ppo.gae(rw, va, dn, pol.value(obs).detach())
        flat = lambda xs: torch.stack(xs).reshape(T * 2, *xs[0].shape[1:])  # noqa: E731
        batch = ppo.Rollout(
            flat(ob), flat(th), flat(pr), flat(lp), adv.reshape(-1), ret.reshape(-1), va.reshape(-1)
        )
        stats = ppo.update(pol, opt, batch, epochs=2, minibatches=2)
        assert np.isfinite(list(stats.values())).all()
    assert any(not torch.equal(b, p) for b, p in zip(before, pol.parameters(), strict=True))
    for e in envs:
        e.close()
