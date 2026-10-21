"""Animation QA (B10): measurements on a baked episode, each with its limit. ``check`` returns {name: (value, limit,
ok)}; ``report`` formats them. Nothing here changes the animation: it only measures what the baker produced."""

from __future__ import annotations

import math

import numpy as np

from .. import constants as C
from . import camera, rig
from .kin import FPS, tick_to_frame
from .track import ik_smooth_air

LIMITS = {
    "foot_drift_cm": 1.0, "foot_below_floor_cm": 0.5, "stance_foot_height_cm": 1.0, "bone_length_mm": 0.1,
    "root_xy_mm": 1.0, "root_jump_mm": 1.0, "catch_ball_cm": 3.0, "catch_gap_err_cm": 2.0, "intercept_cm": 5.0,
    "release_jump_cm": 2.0, "pose_pop_deg": 25.0, "overlap_cm": 4.0, "cut_in_window": 0, "pan_deg_s": 60.0,
    "knee_deg": (0.0, 150.0), "elbow_deg": (0.0, 155.0), "hip_deg": (-20.0, 120.0),
}  # fmt: skip


def _angle(a, b):
    c = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0], strict=True))


def joint_angles(H: float, pelvis, q):
    """Knee / elbow / hip angles (degrees) of both sides for one frame."""
    pos, rot = rig.fk(H, pelvis, q)
    out = {"knee": [], "elbow": [], "hip": []}
    for s in "LR":
        th, sh, ft = (pos[rig.J[f"{n}_{s}"]] for n in ("thigh", "shin", "foot"))
        out["knee"].append(_angle(sh - th, ft - sh))
        ua, fa, ha = (pos[rig.J[f"{n}_{s}"]] for n in ("upper_arm", "forearm", "hand"))
        out["elbow"].append(_angle(fa - ua, ha - fa))
        d = (sh - th) / np.linalg.norm(sh - th)
        local = rot[0].T @ d
        out["hip"].append(math.degrees(math.atan2(local[0], -local[2])))
    return out, pos


def check(baked, direction: camera.Direction | None = None, stride: int = 4) -> dict:
    sim, plan = baked.sim, baked.plan
    K = sim.K
    res: dict[str, tuple] = {}

    def put(name, value, limit, ok=None):
        if ok is None:
            ok = value <= limit
        res[name] = (float(value), limit, bool(ok))

    drift = floor_below = stance_h = bone = root_xy = root_jump = pop = 0.0
    ranges = {"knee": [999.0, -999.0], "elbow": [999.0, -999.0], "hip": [999.0, -999.0]}
    for cid, tr in baked.tracks.items():
        k = baked.kins[cid]
        H = k.H
        air_w = ik_smooth_air(k.air)
        ground = air_w == 0.0
        sole = tr["ankle"][:, :, 2] - rig.ANKLE_Z * H
        floor_below = max(floor_below, float(-sole.min()) * 100)
        for i in range(2):
            for s, e in _runs(k.feet["stance"][:, i] & ground):
                seg = tr["ankle"][s:e, i]
                if len(seg) > 2:
                    drift = max(drift, float(np.linalg.norm(seg - seg[0], axis=1).max()) * 100)
                    stance_h = max(stance_h, float(sole[s:e, i].max()) * 100)
        root_xy = max(
            root_xy,
            float(np.linalg.norm(tr["pelvis"][:, :2] - sim.pos[cid][:, :2], axis=1).max()) * 1000,
        )
        air = k.air & (air_w >= 1.0)
        if air.any():
            expect = k.jz + rig.PELVIS_REST_Z * H - 0.02 * H
            root_jump = max(
                root_jump, float(np.abs(tr["pelvis"][air, 2] - expect[air]).max()) * 1000
            )
        for f in range(0, K, stride * 3):
            ang, pos = joint_angles(H, tr["pelvis"][f], tr["q"][f])
            for name, vals in ang.items():
                ranges[name][0] = min(ranges[name][0], min(vals))
                ranges[name][1] = max(ranges[name][1], max(vals))
            for j in range(1, 21):
                bone = max(
                    bone,
                    abs(
                        float(np.linalg.norm(pos[j] - pos[rig.PAR[j]]))
                        - float(np.linalg.norm(rig.OFFSET[j])) * H
                    )
                    * 1000,
                )
        q = tr["q"][::stride]
        dots = np.abs(np.sum(q[1:] * q[:-1], axis=-1)).clip(0, 1)
        pop = max(pop, float(np.degrees(2 * np.arccos(dots)).max()) / 1.0)  # per 60 Hz frame
    put("foot_drift_cm", drift, LIMITS["foot_drift_cm"])
    put("foot_below_floor_cm", floor_below, LIMITS["foot_below_floor_cm"])
    put("stance_foot_height_cm", stance_h, LIMITS["stance_foot_height_cm"])
    put("bone_length_mm", bone, LIMITS["bone_length_mm"])
    put("root_xy_mm", root_xy, LIMITS["root_xy_mm"])
    put("root_jump_mm", root_jump, LIMITS["root_jump_mm"])
    put("pose_pop_deg", pop, LIMITS["pose_pop_deg"])
    for name in ("knee", "elbow", "hip"):
        lo, hi = LIMITS[f"{name}_deg"]
        v = ranges[name]
        put(
            f"{name}_deg",
            max(lo - v[0], v[1] - hi, 0.0),
            0.0,
            ok=v[0] >= lo - 0.5 and v[1] <= hi + 0.5,
        )
        res[f"{name}_deg"] = (
            float(max(lo - v[0], v[1] - hi, 0.0)),
            f"within {lo}..{hi} (saw {v[0]:.0f}..{v[1]:.0f})",
            v[0] >= lo - 0.5 and v[1] <= hi + 0.5,
        )
    # --- contact frames
    if plan.f_catch is not None and baked.outcome in ("DUNK", "CATCH_NO_DUNK"):
        tr = baked.tracks["dunker"]
        pl, pr = tr["palm"][plan.f_catch]
        ball = baked.ball[plan.f_catch]
        put(
            "catch_ball_cm",
            float(np.linalg.norm((pl + pr) / 2 - ball)) * 100,
            LIMITS["catch_ball_cm"],
        )
        put(
            "catch_gap_err_cm",
            abs(float(np.linalg.norm(pl - pr)) - 0.24) * 100,
            LIMITS["catch_gap_err_cm"],
        )
    if plan.f_intercept is not None and plan.interceptor:
        tr = baked.tracks[plan.interceptor]
        pl, pr = tr["palm"][plan.f_intercept]
        ball = baked.ball[plan.f_intercept]
        d = min(float(np.linalg.norm(pl - ball)), float(np.linalg.norm(pr - ball))) - C.BALL_R
        put("intercept_cm", d * 100, LIMITS["intercept_cm"])
    if plan.f_release is not None:
        f0 = plan.f_release
        seg = baked.ball[max(f0 - 8, 1) : f0 + 8]
        step = np.diff(seg, axis=0)
        jerk = np.linalg.norm(np.diff(step, axis=0), axis=1)
        put("release_jump_cm", float(jerk.max()) * 100, LIMITS["release_jump_cm"])
    # --- characters never interpenetrate beyond the soft-contact budget (except blocks / contests)
    names = [a["id"] for a in baked.meta["actors"]]
    worst = 0.0
    skip = [
        tick_to_frame(e["tick"])
        for e in baked.meta["events"]
        if e["type"] in ("block", "intercept")
    ]
    for f in range(0, K, 6):
        if any(abs(f - s) < FPS * 0.4 for s in skip):
            continue
        p = np.array([sim.pos[n][f, :2] for n in names])
        dm = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(len(p)) * 99
        worst = max(worst, 2 * C.PLAYER_CAPSULE_R - float(dm.min()))
    put("overlap_cm", max(worst, 0.0) * 100, LIMITS["overlap_cm"])
    # --- camera
    if direction is not None:
        bad = sum(1 for c in direction.cuts for a, b in direction.protected if a < c < b)
        put("cut_in_window", bad, 0)
        vt = camera.video_times(direction.slow, K)
        near = np.clip(np.round(vt * FPS).astype(int), 0, K - 1)
        dirs = direction.target[near] - direction.pos[near]
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        cut_set = set(int(np.searchsorted(near, c)) for c in direction.cuts)
        rate = 0.0
        for v in range(1, len(near)):
            if v in cut_set or v - 1 in cut_set:
                continue
            ang = math.degrees(math.acos(float(np.clip(np.dot(dirs[v], dirs[v - 1]), -1, 1))))
            rate = max(rate, ang * 60.0)
        put("pan_deg_s", rate, LIMITS["pan_deg_s"])
    return res


def report(res: dict) -> str:
    lines = []
    for name, (v, lim, ok) in res.items():
        lines.append(f"{'PASS' if ok else 'FAIL'}  {name:24s} {v:9.3f}  (limit {lim})")
    return "\n".join(lines)


def failures(res: dict) -> list[str]:
    return [n for n, (_, _, ok) in res.items() if not ok]
