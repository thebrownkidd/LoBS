"""Characters and ball for one episode: facing, the ball's visual path, arm layers and full pose assembly.

The sim's ball is physical until something visual takes over (hands before release, a catch, a dunk, a deflection);
this module builds the ball the audience sees, then poses every character around it."""

from __future__ import annotations

import math

import numpy as np

from .. import constants as C
from . import ik, kin, pose, rig
from .kin import FPS, Kin, Sim, smooth_step, tick_to_frame


# ------------------------------------------------------------------ helpers
def rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def pelvis_base(k: Kin, f: int) -> np.ndarray:
    return np.array([*k.root_xy[f], k.jz[f] + rig.PELVIS_REST_Z * k.H - k.crouch[f]])


def chest_hold(k: Kin, f: int) -> np.ndarray:
    return pelvis_base(k, f) + rz(k.yaw[f]) @ np.array([0.22 * k.H, 0, 0.17 * k.H])


def overhead_hold(k: Kin, f: int) -> np.ndarray:
    return pelvis_base(k, f) + rz(k.yaw[f]) @ np.array([0.08 * k.H, 0, 0.62 * k.H])


def shoulder_centre(k: Kin, f: int) -> np.ndarray:
    return pelvis_base(k, f) + rz(k.yaw[f]) @ np.array([0.0, 0.0, 0.285 * k.H])


def clamp_reach(k: Kin, f: int, p: np.ndarray, one_hand: bool = False) -> np.ndarray:
    """Nearest point to ``p`` that the character can reach with its palms (a sphere around the shoulders)."""
    c = shoulder_centre(k, f)
    rr = (0.378 * 0.97 if one_hand else 0.34) * k.H
    rr += 0.085 * k.H * ik.smoothstep((p[2] - c[2]) / (0.4 * k.H))
    d = p - c
    n = float(np.linalg.norm(d))
    lo = 0.20 * k.H  # never inside the torso: hands there make the arm IK degenerate
    if n < lo:
        d = d / n * lo if n > 1e-6 else rz(k.yaw[f]) @ np.array([lo, 0.0, 0.0])
        return c + d
    return p if n <= rr else c + d * (rr / n)


def evt(sim: Sim, kind, actor=None):
    e = sim.event(kind, actor)
    return None if e is None else e["tick"]


# ------------------------------------------------------------------ facing
def facing_targets(sim: Sim, actor: dict, v0_dir: float | None) -> np.ndarray:
    """Heading each character wants (radians), before the spring smooths it."""
    name = actor["id"]
    pos = sim.pos
    xy = pos[name][:, :2]
    ball = pos["ball"]
    rel = evt(sim, "release")
    f_rel = tick_to_frame(rel) if rel is not None else None
    f_wind = tick_to_frame(evt(sim, "windup")) if rel is not None else None

    def toward(p):
        d = p[:, :2] - xy
        return np.arctan2(d[:, 1], d[:, 0])

    if actor["role"] == "dunker":
        v = sim.vel[name][:, :2]
        sp = np.linalg.norm(v, axis=1)
        des = np.arctan2(v[:, 1], v[:, 0])
        return kin.ffill_angle(des, sp > 0.5, math.atan2(-xy[0, 1], -xy[0, 0]))
    if actor["role"] == "passer":
        des = toward(pos["dunker"]).copy()
        if f_rel is not None and v0_dir is not None:
            des[max(f_wind - int(0.1 * FPS), 0) : f_rel + int(0.55 * FPS)] = v0_dir
        return des
    # defenders: face the ball holder, then the ball
    des = toward(pos["passer"]).copy()
    if f_rel is not None:
        des[f_rel:] = toward(ball)[f_rel:]
    return des


# ------------------------------------------------------------------ ball
def _bounce(p, v, n, restitution=0.6, drift=(0.35, 0.2)):
    """Drag-free ball with floor bounces for ``n`` frames from state (p, v). Returns positions (n,3)."""
    out = np.zeros((n, 3))
    dt = 1.0 / FPS
    p, v = p.copy(), v.copy()
    for i in range(n):
        v[2] -= C.G * dt
        p = p + v * dt
        if p[2] < C.BALL_R:
            p[2] = C.BALL_R + (C.BALL_R - p[2])
            v[2] = -v[2] * restitution
            v[:2] = v[:2] * 0.8 + np.array(drift) * (0.0 if abs(v[2]) < 0.5 else 1.0) * 0.2
            if abs(v[2]) < 0.4:
                v[2] = 0.0
                p[2] = C.BALL_R
                v[:2] *= 0.98
        out[i] = p
    return out


class BallPlan:
    """What the ball does and who has it when, for the arm layers and the camera."""

    def __init__(self):
        self.pos = None
        self.carry: list = []  # (f0, f1, holder id, mode)  mode: chest | overhead | dunk
        self.f_release = self.f_windup = self.f_catch = self.f_slam = self.f_intercept = None
        self.interceptor = None
        self.pass_dir = None
        self.catch_target = None
        self.intercept_target = None
        self.windup_pts = None


def plan_ball(sim: Sim, kins: dict, outcome: str) -> BallPlan:
    K = sim.K
    bp = BallPlan()
    ball = sim.pos["ball"].copy()
    rel = evt(sim, "release")
    passer = kins["passer"]
    out = ball.copy()
    holds = []
    if rel is None:  # never thrown: the passer keeps it
        for f in range(K):
            out[f] = chest_hold(passer, f)
        bp.pos, bp.carry = out, [(0, K, "passer", "chest")]
        return bp
    f_rel, f_wind = tick_to_frame(rel), tick_to_frame(evt(sim, "windup"))
    bp.f_release, bp.f_windup = f_rel, f_wind
    v0 = np.array(next(e for e in sim.events if e["type"] == "release")["data"]["v0"])
    bp.pass_dir = math.atan2(v0[1], v0[0])
    # --- pass: chest -> behind the head -> the sim's hold point exactly at release
    for f in range(0, f_wind):
        out[f] = chest_hold(passer, f)
    p2 = ball[f_rel].copy()
    mid = (f_wind + f_rel) // 2
    # up and over the head (never through it): chest -> just above and behind the head -> the sim's hold point
    head_mid = pelvis_base(passer, mid) + rz(passer.yaw[mid]) @ np.array(
        [-0.06 * passer.H, 0.0, 0.60 * passer.H]
    )
    p0 = chest_hold(passer, f_wind)
    n = f_rel - f_wind
    T = n / FPS
    v_end = (
        0.8 * v0
    )  # arrive moving with the throw (the sim's release impulse then adds only the last 20 %)
    c2 = p2 - v_end * (T / 3.0)
    c1 = (
        8.0 * head_mid - p0 - p2
    ) / 3.0 - c2  # exact: the cubic passes head_mid at its half-way point
    for i, f in enumerate(range(f_wind, f_rel + 1)):
        t = i / max(n, 1)
        out[f] = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t**2 * c2 + t**3 * p2
    bp.windup_pts = (p0, head_mid, p2)
    holds.append((0, f_rel, "passer", "chest"))
    # --- what happens to the ball after the release
    f_catch = (
        tick_to_frame(evt(sim, "catch", "dunker"))
        if evt(sim, "catch", "dunker") is not None
        else None
    )
    f_int = None
    for e in sim.events:
        if e["type"] == "intercept":
            f_int, bp.interceptor = tick_to_frame(e["tick"]), e["actor"]
            break
    dk = kins["dunker"]
    if outcome in ("DUNK", "CATCH_NO_DUNK") and f_catch is not None:
        bp.f_catch = f_catch
        target = clamp_reach(dk, f_catch, ball[f_catch])
        bp.catch_target = target
        lead = int(0.25 * FPS)
        for f in range(max(f_rel, f_catch - lead), f_catch + 1):
            w = smooth_step((f - (f_catch - lead)) / lead)
            out[f] = ball[f] + w * (target - ball[f_catch])
        carry_f = f_catch
        # who carries it, how: overhead while airborne, at the chest on the ground
        slam = None
        if outcome == "DUNK":
            for s, e in zip(dk.takeoffs, dk.lands, strict=True):
                if e > f_catch:
                    apex = s + int(np.argmax(dk.jz[s:e]))
                    slam = max(apex, f_catch + int(0.12 * FPS))
                    break
        bp.f_slam = slam
        end_carry = slam if slam is not None else K
        for f in range(carry_f + 1, end_carry):
            anchor = overhead_hold(dk, f) if dk.air[f] else chest_hold(dk, f)
            w = smooth_step((f - carry_f) / (0.15 * FPS))
            out[f] = (1 - w) * target + w * anchor
        holds.append(
            (
                f_catch,
                end_carry,
                "dunker",
                "overhead" if dk.air[min(end_carry - 1, K - 1)] else "chest",
            )
        )
        if slam is not None:
            lead_s = int(0.12 * FPS)
            start = out[max(slam - lead_s, carry_f)]
            w_top = clamp_reach(dk, slam, np.array([0.0, 0.0, 3.30]), one_hand=False)
            for f in range(max(slam - lead_s, carry_f), slam):
                w = smooth_step((f - (slam - lead_s)) / lead_s)
                out[f] = (1 - w) * start + w * w_top
            holds[-1] = (f_catch, slam, "dunker", "dunk")
            # push over the hoop, fall through the net, bounce on the floor
            top = np.array([0.0, 0.0, max(w_top[2], 3.20)])
            push = int(0.10 * FPS)
            for f in range(slam, min(K, slam + push)):
                w = smooth_step((f - slam) / push)
                out[f] = (1 - w) * w_top + w * np.array([0.0, 0.0, top[2] - 0.05])
            if slam + push < K:
                out[slam + push :] = _bounce(
                    np.array([0.0, 0.0, top[2] - 0.05]), np.array([0.0, 0.0, -2.0]), K - slam - push
                )
        elif f_catch is not None:
            pass
        bp.carry = holds
        if slam is None and carry_f + 1 < K:
            pass
    elif outcome == "INTERCEPTED" and f_int is not None:
        bp.f_intercept = f_int
        d = kins[bp.interceptor]
        target = clamp_reach(d, f_int, ball[f_int], one_hand=True)
        bp.intercept_target = target
        lead = int(0.10 * FPS)
        for f in range(max(f_rel, f_int - lead), f_int + 1):
            w = smooth_step((f - (f_int - lead)) / lead)
            out[f] = ball[f] + w * (target - ball[f_int])
        v_in = sim.vel["ball"][f_int]
        v_out = np.array([-0.3 * v_in[0], -0.3 * v_in[1], 3.0])
        if f_int + 1 < K:
            out[f_int + 1 :] = _bounce(target, v_out, K - f_int - 1)
        bp.carry = holds
    else:
        bp.carry = holds
    bp.pos = out
    return bp


def ball_orientation(sim: Sim, bp: BallPlan) -> np.ndarray:
    """(K,4) quaternions: 2.5 rev/s backspin while in the air, frozen while held."""
    K = sim.K
    q = np.tile(rig.IDENT, (K, 1))
    cur = rig.IDENT.copy()
    held = np.zeros(K, bool)
    for f0, f1, _, _ in bp.carry:
        held[f0:f1] = True
    v = np.gradient(bp.pos, axis=0) * FPS
    for f in range(K):
        sp = float(np.hypot(v[f, 0], v[f, 1]))
        if not held[f] and sp > 0.8:
            axis = np.cross([0, 0, 1.0], [v[f, 0], v[f, 1], 0.0])
            axis = axis / max(np.linalg.norm(axis), 1e-9)
            cur = rig.qnorm(rig.qmul(rig.qaxis(axis, 2.5 * 2 * math.pi / FPS), cur))
        q[f] = cur
    return q


# ------------------------------------------------------------------ arms
def _arm_palm(
    k: Kin, f: int, side: int, swing: float, elbow: float, yaw: float, base: np.ndarray
) -> np.ndarray:
    """Palm for a swinging arm: upper arm swung ``swing`` rad forward, forearm bent ``elbow`` further forward."""
    s = np.array([0.0, side * 0.115, 0.285]) * k.H
    d1 = np.array([math.sin(swing), 0.0, -math.cos(swing)])
    d2 = np.array([math.sin(swing + elbow), 0.0, -math.cos(swing + elbow)])
    local = s + d1 * rig.ARM_L1 * k.H + d2 * (rig.ARM_L2 + rig.PALM) * k.H
    return base + rz(yaw) @ local


def _arm_up(k: Kin, side: int, f: int, base, yaw, level="overhead"):
    s = np.array([0.0, side * 0.115, 0.285]) * k.H
    off = (
        np.array([0.10, side * 0.02, 0.36])
        if level == "overhead"
        else np.array([0.20, side * 0.03, 0.06])
    )
    return base + rz(yaw) @ (s + off * k.H)


def base_palms(k: Kin, f: int, role_hands_up: bool, air_w: float = 0.0):
    """Default arms: swing opposite the legs when running, hands up for defenders at rest, relaxed otherwise."""
    base = pelvis_base(k, f)
    yaw = k.yaw[f]
    v = k.speed[f]
    if k.team == "defense" and air_w > 0.0:
        base_l, base_r = base_palms(k, f, role_hands_up, 0.0)
        up_l, up_r = _arm_up(k, +1, f, base, yaw), _arm_up(k, -1, f, base, yaw)
        return base_l * (1 - air_w) + up_l * air_w, base_r * (1 - air_w) + up_r * air_w
    if k.team == "defense" and v < 2.5:
        lvl = "overhead" if role_hands_up else "shoulder"
        return _arm_up(k, +1, f, base, yaw, lvl), _arm_up(k, -1, f, base, yaw, lvl)
    if v > 0.6:
        head = np.array([math.cos(yaw), math.sin(yaw)])
        fwd_l = float(np.dot(k.feet["foot"][f, 0, :2] - k.root_xy[f], head))
        amp = math.radians(35.0) * min(v / 5.0, 1.0)
        sw = np.clip(fwd_l / (0.5 * k.H), -1.0, 1.0) * amp
        e = math.radians(20.0 + 70.0 * min(max(v - 1.0, 0.0) / 3.0, 1.0))
        # the left arm swings with the right leg
        return _arm_palm(k, f, +1, +sw * 0 - sw, e, yaw, base), _arm_palm(
            k, f, -1, +sw, e, yaw, base
        )
    return _arm_palm(k, f, +1, 0.0, math.radians(15), yaw, base), _arm_palm(
        k, f, -1, 0.0, math.radians(15), yaw, base
    )


def two_hand_targets(ball_p: np.ndarray, yaw_dir: float, gap: float = 0.24):
    """Palm centres on either side of the ball, perpendicular to the direction it is held toward."""
    lat = np.array([-math.sin(yaw_dir), math.cos(yaw_dir), 0.0])
    return ball_p + lat * gap / 2, ball_p - lat * gap / 2


def build_arm_layers(sim, kins, bp: BallPlan, outcome: str):
    """Per character a function f -> (palmL, palmR, weight) overrides, evaluated by ``assemble``."""
    layers = {k: [] for k in kins}
    ball = bp.pos
    passer = kins["passer"]
    dk = kins["dunker"]
    sec = lambda s: int(s * FPS)  # noqa: E731

    # --- passer: both hands on the ball until the follow-through, then arms extend and relax
    def passer_layer(f):
        end = (bp.f_release or 0) + sec(0.35)
        if bp.f_release is None or f <= bp.f_release:
            return two_hand_targets(ball[f], passer.yaw[f]), 1.0
        if f <= end:
            t = (f - bp.f_release) / max(end - bp.f_release, 1)
            reach = ball[bp.f_release] + np.array(
                [math.cos(bp.pass_dir), math.sin(bp.pass_dir), 0.25]
            ) * (0.35 * (1 - t) + 0.05)
            left, right = two_hand_targets(reach, bp.pass_dir)
            return (left, right), 1.0 - 0.7 * smooth_step(t)
        return None, 0.0

    layers["passer"].append(passer_layer)

    # --- dunker: reach for the ball, carry it, dunk it
    def dunker_layer(f):
        if bp.f_catch is not None:
            f_start = bp.f_catch - sec(0.30)
            if f_start <= f <= bp.f_catch:
                w = smooth_step((f - f_start) / sec(0.25))
                return two_hand_targets(ball[f], dk.yaw[f] if not dk.air[f] else dk.yaw[f]), w
            for f0, f1, who, mode in bp.carry:
                if who == "dunker" and f0 < f <= f1 + sec(0.30):
                    w = 1.0 if f <= f1 else 1.0 - smooth_step((f - f1) / sec(0.30))
                    if mode == "dunk" and f > f1:
                        rim = np.array([0.1, 0.0, 3.15])
                        return (rim + np.array([0, 0.1, 0]), rim - np.array([0, 0.1, 0])), w
                    return two_hand_targets(ball[min(f, f1)], dk.yaw[f]), w
        for s0, e0 in zip(dk.takeoffs, dk.lands, strict=True):
            if s0 - sec(0.12) <= f < e0 + sec(0.15) and (
                bp.f_catch is None or f < bp.f_catch - sec(0.30)
            ):
                w = (
                    0.85
                    * smooth_step((f - (s0 - sec(0.12))) / sec(0.22))
                    * (1.0 - smooth_step((f - e0) / sec(0.15)))
                )
                up = (
                    _arm_up(dk, +1, f, pelvis_base(dk, f), dk.yaw[f]),
                    _arm_up(dk, -1, f, pelvis_base(dk, f), dk.yaw[f]),
                )
                return up, w
        return None, 0.0

    layers["dunker"].append(dunker_layer)

    # --- defenders: one hand to the ball at the interception, both up in the air
    if bp.f_intercept is not None and bp.interceptor in kins:
        d = kins[bp.interceptor]

        def int_layer(f):
            f0, f1 = bp.f_intercept - sec(0.25), bp.f_intercept + sec(0.15)
            if f0 <= f <= f1:
                w = smooth_step((f - f0) / sec(0.2)) * (
                    1.0 - smooth_step((f - bp.f_intercept) / sec(0.15))
                )
                hand = bp.intercept_target
                right = pelvis_base(d, f) + rz(d.yaw[f]) @ np.array([0.1, -0.2, 0.9]) * d.H
                other = _arm_up(d, +1, f, pelvis_base(d, f), d.yaw[f])
                pl, pr = (
                    (other, hand)
                    if np.dot(hand - right, rz(d.yaw[f])[:, 1]) < 0
                    else (hand, _arm_up(d, -1, f, pelvis_base(d, f), d.yaw[f]))
                )
                return (pl, pr), w
            return None, 0.0

        layers[bp.interceptor].append(int_layer)
    return layers


# ------------------------------------------------------------------ assembly
# joints whose rotation rate is limited: arms, shoulders, spine, neck and head (never the legs: planted feet stay exact)
UPPER_BODY = tuple(
    rig.J[n]
    for n in rig.JOINTS
    if n.startswith(
        (
            "spine",
            "chest",
            "neck",
            "head",
            "clavicle",
            "upper_arm",
            "forearm",
            "hand",
            "foot",
            "toe",
        )
    )
)
MAX_STEP_DEG = 6.0  # per 240 Hz frame = 24 degrees per 60 Hz video frame
LEG_JOINTS = tuple(rig.J[n] for n in rig.JOINTS if n.startswith(("thigh", "shin", "foot", "toe")))
LEG_STEP_DEG = 6.0  # the legs are only limited where the IK has a discontinuity


def slew_limit(q: np.ndarray, joints, max_step: float) -> np.ndarray:
    """Limit how far each listed joint may rotate per frame (a forward pass followed by a backward pass, so the
    limiter anticipates a fast move instead of only trailing it)."""
    out = q.copy()
    for order in (
        range(len(q) - 2, -1, -1),
        range(1, len(q)),
    ):  # backward first: errors land before a fast move
        prev = None
        for f in order:
            if prev is None:
                prev = f - 1 if f > 0 and order.step > 0 else f + 1
            for j in joints:
                a, b = out[prev][j], out[f][j]
                d = abs(float(np.dot(a, b)))
                ang = 2.0 * math.acos(min(max(d, 0.0), 1.0))
                if ang > max_step:
                    out[f][j] = rig.slerp(a, b if np.dot(a, b) >= 0 else -b, max_step / ang)
            prev = f
    return out


def assemble(sim: Sim, k: Kin, bp: BallPlan, arm_layers: list, hands_up: bool, look_fn):
    """Pose one character on every frame. Returns dict of arrays: pelvis (K,3), q (K,21,4), plus QA data."""
    K = sim.K
    H = k.H
    pel = np.zeros((K, 3))
    qs = np.zeros((K, 21, 4))
    ankle = np.zeros((K, 2, 3))
    palm = np.zeros((K, 2, 3))
    air_w = ik_smooth_air(k.air)
    for f in range(K):
        yaw = k.yaw[f]
        base = np.array(
            [*k.root_xy[f], k.jz[f] + rig.PELVIS_REST_Z * H + k.pelvis_dz[f] - k.crouch[f]]
        )
        if (
            air_w[f] > 0.0
        ):  # blend the pelvis between its ground height (with crouch) and its airborne height
            air_z = k.jz[f] + rig.PELVIS_REST_Z * H - 0.02 * H
            base[2] = (1.0 - air_w[f]) * base[2] + air_w[f] * air_z
        pelvis_q = pose.euler_q(yaw + k.counter[f], k.lean[f], k.roll[f])
        # feet: gait targets blended with a tucked pose while airborne
        feet = []
        for i in range(2):
            gt = k.feet["foot"][f, i]
            # tucked legs hang from the actual hips (pelvis frame), so the pelvis never has to be pulled down
            hip_local = np.array([0.0, (C.HIP_HALF if i == 0 else -C.HIP_HALF) * H, 0.0])
            tuck = base + rig.qmat(pelvis_q) @ (hip_local + np.array([0.06 * H, 0.0, -0.43 * H]))
            g_ankle = np.array(
                [gt[0], gt[1], gt[2] + rig.ANKLE_Z * H + (k.jz[f] if not k.air[f] else 0.0)]
            )
            if air_w[f] > 0:
                g_ankle = (1 - air_w[f]) * g_ankle + air_w[f] * tuck
                g_ankle[2] = max(g_ankle[2], rig.ANKLE_Z * H)  # a foot never goes through the floor
            su = k.feet["swing_u"][f, i]
            pitch = 0.0
            if su >= 0:
                pitch = math.radians(25.0) * (1 - smooth_step(su / 0.25)) - math.radians(
                    15.0
                ) * smooth_step((su - 0.75) / 0.25)
            feet.append((g_ankle, k.feet["foot_yaw"][f, i], pitch))
        pl = pr = None
        palms = base_palms(k, f, hands_up, air_w[f])
        w_tot = 0.0
        for layer in arm_layers:
            tg, w = layer(f)
            if tg is not None and w > 0:
                palms = (palms[0] * (1 - w) + tg[0] * w, palms[1] * (1 - w) + tg[1] * w)
                w_tot = max(w_tot, w)
        pl, pr = palms
        hy, hp = look_fn(f, base + np.array([0, 0, 0.37 * H]), yaw)
        chest_q = pose.euler_q(-0.5 * k.counter[f], 0.0, 0.0)
        q, pelv, info = pose.solve_pose(H, {"pelvis_pos": base, "pelvis_q": pelvis_q, "chest_q": chest_q,
                                            "feet": feet, "arms": (pl, pr), "head": (hy, hp)})  # fmt: skip
        pel[f], qs[f] = pelv, q
    qs = slew_limit(qs, UPPER_BODY, math.radians(MAX_STEP_DEG))
    for f in range(K):  # measure the skeleton that is actually rendered
        pos, rot = rig.fk(H, pel[f], qs[f])
        ankle[f] = [pos[rig.J["foot_L"]], pos[rig.J["foot_R"]]]
        palm[f] = [
            pos[rig.J[f"hand_{s}"]] + rot[rig.J[f"hand_{s}"]] @ np.array([0, 0, -rig.PALM * H])
            for s in "LR"
        ]
    return {"pelvis": pel, "q": qs, "ankle": ankle, "palm": palm}


def ik_smooth_air(air: np.ndarray) -> np.ndarray:
    """0..1 weight of the airborne (tucked) pose: ramps in/out over 0.06 s around each jump."""
    w = air.astype(float)
    n = int(0.06 * FPS)
    ext = np.zeros_like(w)
    for s, e in kin._segments(air):
        for k in range(max(0, s - n), s):
            ext[k] = max(ext[k], (k - (s - n)) / n)
        for k in range(e, min(len(w), e + n)):
            ext[k] = max(ext[k], 1.0 - (k - e) / n)
    return np.maximum(w, ext)
