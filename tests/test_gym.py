import numpy as np
import pytest

import lobs
from lobs import props, rel

gymnasium = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env  # noqa: E402

BACKENDS = lobs.backends.available()


def cart_world():
    """A pendulum on a motor: observe the angle and tip height; reward and termination are ours."""
    w = lobs.World("pend", dt=0.002, substeps=5)
    p = w.add_actor("pend", root="fixed", pos=(0, 0, 2))
    base = p.add_body("base", geoms=[lobs.Box(size=(0.1, 0.1, 0.1))], mass=1.0, collision_group=3)
    base.add_child("rod", pos=(0, 0, -0.5), joint=lobs.Hinge("hinge", axis=(0, 1, 0), pos=(0, 0, 0.5), damping=0.05),
                   geoms=[lobs.Capsule(radius=0.02, length=1.0)], mass=0.5, collision_group=3)  # fmt: skip
    w.spec.contact.disabled_group_pairs.append((3, 3))
    p.add_actuator("motor", "hinge", kind="effort", ctrl_range=(-3.0, 3.0))
    p.set_initial(q={"hinge": 0.4})
    th = w.measure("theta", props.joint_position(p.joint("hinge")))
    w.measure("omega", props.joint_velocity(p.joint("hinge")))
    w.measure("tip", props.position(p.body("rod")))
    w.measure("upright", th.eq(0.0, tol=0.05))
    return w


@pytest.mark.parametrize("be", BACKENDS)
def test_spaces_step_reset_and_user_defined_task(be):
    env = lobs.GymEnv(cart_world(), be, observe=["theta", "omega", "tip"], reward=lambda sim: -abs(sim["theta"]),
                      terminated=lambda sim: abs(sim["theta"]) > 1.0, max_seconds=1.0)  # fmt: skip
    assert env.observation_space.shape == (5,) and env.observation_names == [
        "theta",
        "omega",
        "tip[0]",
        "tip[1]",
        "tip[2]",
    ]
    assert (
        env.action_space.shape == (1,)
        and env.action_space.low[0] == -3
        and env.action_space.high[0] == 3
    )
    obs, info = env.reset(seed=1)
    assert (
        obs.dtype == np.float32 and obs[0] == pytest.approx(0.4, abs=1e-6) and info["time"] == 0.0
    )
    total, steps, trunc = 0.0, 0, False
    while not trunc:
        obs, r, term, trunc, info = env.step(np.array([0.0]))
        total += r
        steps += 1
        assert not term
    assert (
        steps == 100 and info["time"] == pytest.approx(1.0) and total < 0
    )  # 1 s at 10 ms per tick
    assert "theta" in info["measurements"]
    # no task is forced: with no reward / terminated callbacks everything is 0 / False
    bare = lobs.GymEnv(cart_world(), be, observe=["theta"])
    bare.reset()
    _, r, term, trunc, _ = bare.step(np.zeros(1))
    assert r == 0.0 and term is False and trunc is False


@pytest.mark.parametrize("be", BACKENDS)
def test_reward_from_measure_and_dict_obs_and_action_effect(be):
    w = cart_world()
    env = lobs.GymEnv(w, be, observe=["theta", rel.between(w.actor("pend").body("rod"), lobs.WORLD).distance()],
                      reward="theta", obs_as_dict=True)  # fmt: skip
    assert set(env.observation_space.spaces) == {"theta", "__obs0"}
    env.reset(seed=0)
    o_push = [env.step(np.array([3.0]))[0]["theta"][0] for _ in range(20)]
    env.reset(seed=0)
    o_none = [env.step(np.array([0.0]))[0]["theta"][0] for _ in range(20)]
    assert o_push[-1] > o_none[-1]  # positive torque accelerates the angle
    obs, r, *_ = env.step(np.array([0.0]))
    assert r == pytest.approx(obs["theta"][0], abs=1e-5)  # reward given as a measure name


@pytest.mark.parametrize("be", BACKENDS)
def test_gymnasium_env_checker_and_reproducible_reset(be):
    env = lobs.GymEnv(
        cart_world(), be, observe=["theta", "omega"], reward=lambda s: 1.0, max_seconds=0.5
    )
    check_env(env, skip_render_check=True)
    a = env.reset(seed=5)[0]
    for _ in range(10):
        env.step(env.action_space.sample())
    b = env.reset(seed=5)[0]
    assert np.array_equal(a, b)


def test_errors_are_clear():
    with pytest.raises(lobs.errors.LobsError, match="unknown measure"):
        lobs.GymEnv(cart_world(), "mujoco", observe=["nope"])
    env = lobs.GymEnv(cart_world(), "mujoco", observe=["theta"])
    env.reset()
    with pytest.raises(lobs.errors.LobsError, match="action must have 1"):
        env.step(np.zeros(3))
    sub = lobs.GymEnv(cart_world(), "mujoco", observe=["theta"], actions=[])
    assert sub.action_space.shape == (0,)


def test_normalized_actions_map_onto_the_control_range():
    env = lobs.GymEnv(cart_world(), "mujoco", observe=["theta"], normalize_actions=True)
    assert env.action_space.low[0] == -1 and env.action_space.high[0] == 1
    env.reset()
    env.step(np.array([1.0]))
    assert env.sim.state().joint_effort[0] == pytest.approx(
        3.0
    )  # +1 -> the top of the (-3, 3) range
    env.step(np.array([-0.5]))
    assert env.sim.state().joint_effort[0] == pytest.approx(-1.5)
    unbounded = cart_world()
    unbounded.spec.actors[0].actuators[0].ctrl_range = None
    with pytest.raises(lobs.errors.LobsError, match="finite ctrl_range"):
        lobs.GymEnv(unbounded, "mujoco", observe=["theta"], normalize_actions=True)
