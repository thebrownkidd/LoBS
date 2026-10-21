"""Bake an episode into character poses, ball and camera on the 240 Hz internal timeline, then sample video frames.

pixi run lob-bake --episode runs/lobpass/footage/ep12 --out runs/lobpass/footage/ep12/anim.json
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
from dataclasses import dataclass

import numpy as np

from .. import constants as C
from .. import court
from . import camera, kin, rig, track
from .kin import FPS, Kin, Sim


@dataclass
class Baked:
    sim: Sim
    kins: dict
    tracks: dict
    ball: np.ndarray
    ball_q: np.ndarray
    plan: track.BallPlan
    outcome: str
    meta: dict


def _look_target_fn(sim: Sim, name: str, role: str, bp: track.BallPlan):
    """Head look-at: the ball once it is released, otherwise the person with the ball / the dunker."""
    pos = sim.pos

    def fn(f, head_pos, yaw):
        released = bp.f_release is not None and f >= bp.f_release
        if released:
            tgt = bp.pos[f]
        elif role == "passer":
            tgt = pos["dunker"][f] + np.array([0, 0, 1.5])
        elif role == "dunker":
            tgt = pos["passer"][f] + np.array([0, 0, 1.6])
        else:
            tgt = pos["passer"][f] + np.array([0, 0, 1.6])
        return _ik_look(head_pos, yaw, tgt)

    return fn


def _ik_look(head_pos, yaw, tgt):
    from .ik import look_angles

    return look_angles(head_pos, yaw, tgt)


def bake_episode(arrays: dict, meta: dict) -> Baked:
    sim = kin.load_sim(arrays, meta)
    outcome = meta["outcome"]
    rel = next((e for e in meta["events"] if e["type"] == "release"), None)
    v0_dir = None if rel is None else float(np.arctan2(rel["data"]["v0"][1], rel["data"]["v0"][0]))
    kins: dict[str, Kin] = {}
    for a in meta["actors"]:
        kins[a["id"]] = kin.build_kin(sim, a, track.facing_targets(sim, a, v0_dir))
    bp = track.plan_ball(sim, kins, outcome)
    layers = track.build_arm_layers(sim, kins, bp, outcome)
    tracks = {}
    for a in meta["actors"]:
        k = kins[a["id"]]
        hands_up = a["role"] in ("static", "onball")
        tracks[a["id"]] = track.assemble(
            sim, k, bp, layers[a["id"]], hands_up, _look_target_fn(sim, a["id"], a["role"], bp)
        )
    return Baked(sim, kins, tracks, bp.pos, track.ball_orientation(sim, bp), bp, outcome, meta)


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def _sample(arr: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation of (K, ...) data at fractional internal frame positions ``t``."""
    f0 = np.clip(np.floor(t).astype(int), 0, len(arr) - 1)
    f1 = np.minimum(f0 + 1, len(arr) - 1)
    w = (t - f0).reshape((-1,) + (1,) * (arr.ndim - 1))
    return arr[f0] * (1 - w) + arr[f1] * w


def _sample_quat(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Slerp of (K, ..., 4) quaternion arrays; returns (N, ..., 4)."""
    f0 = np.clip(np.floor(t).astype(int), 0, len(q) - 1)
    f1 = np.minimum(f0 + 1, len(q) - 1)
    w = t - f0
    out = np.empty((len(t),) + q.shape[1:])
    flat_a, flat_b = q[f0].reshape(len(t), -1, 4), q[f1].reshape(len(t), -1, 4)
    o = out.reshape(len(t), -1, 4)
    for n in range(len(t)):
        for j in range(flat_a.shape[1]):
            o[n, j] = rig.slerp(flat_a[n, j], flat_b[n, j], w[n])
    return out


def to_video(baked: Baked, direction: camera.Direction, fps_out: int = 60) -> dict:
    """Sample the internal timeline at video rate through the slow-motion remap."""
    K = baked.sim.K
    vt = camera.video_times(direction.slow, K, fps_out)
    fpos = vt * FPS
    near = np.clip(np.round(fpos).astype(int), 0, K - 1)
    frames = {"sim_time": vt}
    chars = {}
    for cid, tr in baked.tracks.items():
        q = _sample_quat(tr["q"], fpos)  # (N, 21, 4) w,x,y,z
        q = np.concatenate([q[..., 1:], q[..., :1]], axis=-1)  # three.js order x,y,z,w
        chars[cid] = {"pelvis": _sample(tr["pelvis"], fpos), "q": q}
    ball_q = _sample_quat(baked.ball_q, fpos)
    return {
        "frames": frames, "chars": chars, "ball": _sample(baked.ball, fpos),
        "ball_q": np.concatenate([ball_q[..., 1:], ball_q[..., :1]], axis=-1),
        "cam_pos": direction.pos[near], "cam_target": direction.target[near], "cam_fov": direction.fov[near],
        "near": near,
    }  # fmt: skip


def _arc(baked: Baked) -> list:
    """Drag-free predicted arc from the release (dotted line in the overlay)."""
    rel = next((e for e in baked.meta["events"] if e["type"] == "release"), None)
    if rel is None:
        return []
    v0 = np.array(rel["data"]["v0"])
    p0 = np.array(baked.meta["hold"])
    pts = []
    t = 0.0
    while t < 3.0:
        p = p0 + v0 * t + np.array([0, 0, -0.5 * C.G * t * t])
        if p[2] < 0.1 and t > 0.2:
            break
        pts.append([float(x) for x in p])
        t += 0.05
    return pts


def export(
    baked: Baked, direction: camera.Direction, out: str | pathlib.Path, fps_out: int = 60
) -> pathlib.Path:
    v = to_video(baked, direction, fps_out)
    meta, sim = baked.meta, baked.sim
    N = len(v["frames"]["sim_time"])
    vt = v["frames"]["sim_time"]

    def vframe(tick):
        return int(np.searchsorted(vt, tick / kin.SIM_HZ))

    rel = next((e for e in meta["events"] if e["type"] == "release"), None)
    ball_z = baked.ball[:, 2]
    f_rel = baked.plan.f_release
    apex = float(ball_z[f_rel : f_rel + int(2.5 * FPS)].max()) if f_rel is not None else None
    v0 = None if rel is None else rel["data"]["v0"]
    readout = None
    if v0 is not None:
        v0a = np.array(v0)
        readout = {"speed": float(np.linalg.norm(v0a)), "angle": float(np.degrees(np.arctan2(v0a[2], np.hypot(v0a[0], v0a[1])))),
                   "apex": apex}  # fmt: skip
    plan = meta["plan"]
    path = [plan["start"]] + ([plan["corner"]] if plan["corner"] else []) + [plan["takeoff"]]
    data = {
        "format": "lobpass-anim", "version": 1, "fps": fps_out, "frames": N,
        "sim_time": b64(vt),
        "characters": [
            {"id": a["id"], "team": a["team"], "role": a["role"], "H": a["H"],
             "number": {"passer": 7, "dunker": 23}.get(a["id"], 1 + int(a["id"][1:]) if a["id"].startswith("d") and a["id"] != "dunker" else 0),
             "pelvis": b64(v["chars"][a["id"]]["pelvis"]), "quats": b64(v["chars"][a["id"]]["q"])}
            for a in meta["actors"]
        ],
        "ball": {"pos": b64(v["ball"]), "quat": b64(v["ball_q"])},
        "camera": {"pos": b64(v["cam_pos"]), "target": b64(v["cam_target"]), "fov": b64(v["cam_fov"]),
                   "cuts": [int(np.searchsorted(v["near"], c)) for c in direction.cuts]},
        "events": [{"frame": vframe(e["tick"]), "type": e["type"], "actor": e["actor"]} for e in meta["events"]]
        + ([{"frame": vframe(baked.plan.f_slam * kin.SIM_HZ / FPS), "type": "slam", "actor": "dunker"}] if baked.plan.f_slam else []),
        "overlay": {"stage": meta["stage"], "seed": meta["seed"], "outcome": meta["outcome"], "reward": meta["reward"],
                    "checkpoint": meta["checkpoint"], "readout": readout, "plan_path": path, "arc": _arc(baked),
                    "decisions": [{"t": d["tick"] / kin.SIM_HZ, "p": d["p_throw"]} for d in meta["decisions"]],
                    "release_frame": None if rel is None else vframe(rel["tick"]),
                    "captions": meta.get("captions", [])},
        "slow": None if direction.slow is None else list(direction.slow),
        "rig": {"joints": list(rig.JOINTS), "parent": [int(x) for x in rig.PAR], "offset": rig.OFFSET.tolist(),
                "arm": [rig.ARM_L1, rig.ARM_L2, rig.HAND_L], "leg": [rig.LEG_L1, rig.LEG_L2], "ankle": rig.ANKLE_Z},
        "court": {k: [np.asarray(pl).tolist() for pl in v] for k, v in court.line_polylines().items()},
        "constants": {"ball_r": C.BALL_R, "rim_r": C.RIM_RING_R, "rim_z": C.RIM_TUBE_Z, "board": list(C.BOARD_SIZE),
                      "board_center": list(C.BOARD_CENTER), "x0": C.BASELINE_X, "x1": C.HALFCOURT_X, "y": C.SIDELINE_Y,
                      "lane_y": C.LANE_HALF_WIDTH, "ft_x": C.FT_LINE_X, "line_w": C.LINE_WIDTH},
    }  # fmt: skip
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data))
    _ = sim
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = pathlib.Path(args.episode)
    arrays = dict(np.load(d / "episode.npz"))
    meta = json.loads((d / "episode.json").read_text())
    baked = bake_episode(arrays, meta)
    direction = camera.direct(baked)
    out = export(baked, direction, args.out or d / "anim.json")
    print(
        f"baked {baked.sim.K} internal frames x {len(baked.tracks)} characters ({baked.outcome}) -> {out}"
    )


if __name__ == "__main__":
    main()
