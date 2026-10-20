"""Visualization, independent of the physics engine.

* ``Renderer``  - offscreen RGB images via a MuJoCo "visual twin": every body becomes a mocap body posed
  from LoBS state, so Chrono simulations render exactly like MuJoCo ones.
* ``log_rerun`` - log a run (simulation or trajectory) to the Rerun viewer / an .rrd file: geometry once,
  body transforms, anchors and measures per tick.
* ``write_png`` - dependency-free PNG writer.
"""

from __future__ import annotations

import colorsys
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import math3d as m3
from .assets import load_mesh
from .errors import LobsError

_STATIC_RGBA = (0.62, 0.64, 0.68, 1.0)
_PLANE_RGBA = (0.82, 0.83, 0.85, 1.0)
_MESH_TYPES = ("convex_mesh", "convex_set", "triangle_mesh")


@dataclass
class Drawable:
    body: int  # registry-global body index
    kind: str  # box | sphere | cylinder | capsule | plane | mesh
    pos: np.ndarray
    quat: np.ndarray
    rgba: tuple
    size: tuple = ()  # box full extents | (radius,) | (radius, length)
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    label: str = ""

    @property
    def radius(self) -> float:
        """Bounding-sphere radius around the geom centre (planes excluded by callers)."""
        if self.kind == "box":
            return 0.5 * float(np.linalg.norm(self.size))
        if self.kind == "sphere":
            return float(self.size[0])
        if self.kind == "cylinder":
            return float(np.hypot(self.size[0], self.size[1] / 2))
        if self.kind == "capsule":
            return float(self.size[1] / 2 + self.size[0])
        if self.kind == "mesh":
            return float(np.linalg.norm(self.vertices, axis=1).max())
        return 0.0


def _palette(i: int) -> tuple:
    r, g, b = colorsys.hsv_to_rgb((i * 0.381966) % 1.0, 0.55, 0.9)
    return (r, g, b, 1.0)


def drawables(spec, reg) -> list[Drawable]:
    """Everything worth drawing. A body with visual geoms shows those; otherwise its collision geoms."""
    out: list[Drawable] = []
    for ai, actor in enumerate(spec.actors):
        for bspec, gi in zip(actor.bodies, reg.actors[ai].bodies, strict=True):
            info = reg.bodies[gi]
            visual = [g for g in bspec.geoms if g.role in ("visual", "both")]
            geoms = visual or [g for g in bspec.geoms if g.role == "collision"]
            for g in geoms:
                color = (
                    tuple(g.rgba)
                    if g.rgba is not None
                    else (_STATIC_RGBA if info.static else _palette(gi))
                )
                sh = g.shape
                common = dict(
                    body=gi,
                    pos=np.array(g.pose.pos),
                    quat=np.array(g.pose.quat),
                    rgba=color,
                    label=info.path,
                )
                if sh.type == "box":
                    out.append(Drawable(kind="box", size=tuple(sh.size), **common))
                elif sh.type == "sphere":
                    out.append(Drawable(kind="sphere", size=(sh.radius,), **common))
                elif sh.type in ("cylinder", "capsule"):
                    out.append(Drawable(kind=sh.type, size=(sh.radius, sh.length), **common))
                elif sh.type == "plane":
                    out.append(
                        Drawable(
                            kind="plane",
                            **{**common, "rgba": _PLANE_RGBA if g.rgba is None else color},
                        )
                    )
                elif sh.type in _MESH_TYPES:
                    files = list(sh.files) if sh.type == "convex_set" else [sh.file]
                    for f in files:
                        v, fc = load_mesh(reg.base_dir, f)
                        out.append(Drawable(kind="mesh", vertices=v, faces=fc, **common))
    return out


# ---------------------------------------------------------------------------------- offscreen images
class Renderer:
    """Offscreen renderer. ``render(frame)`` draws any StateFrame of this world."""

    def __init__(self, spec, reg, width: int = 640, height: int = 480):
        import mujoco

        self._mj = mujoco
        self.spec, self.reg, self.width, self.height = spec, reg, width, height
        self.items = drawables(spec, reg)
        if not self.items:
            raise LobsError("nothing to draw: the world has no visual or collision geometry")
        ms = mujoco.MjSpec()
        ms.visual.global_.offwidth = max(width, 640)
        ms.visual.global_.offheight = max(height, 480)
        ms.visual.headlight.ambient = [0.45, 0.45, 0.45]
        ms.visual.headlight.diffuse = [0.7, 0.7, 0.7]
        sky = ms.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                             rgb1=[0.86, 0.9, 0.95], rgb2=[0.5, 0.62, 0.78], width=64, height=64)  # fmt: skip
        assert sky is not None
        self._extent_plane = self._plane_half_extent()
        bodies, n_mesh = {}, 0
        for it in self.items:
            if it.body not in bodies:
                info = reg.bodies[it.body]
                if info.static:
                    pos, quat = info.world_pos, info.world_quat
                else:
                    pos, quat = np.zeros(3), m3.IDENTITY_QUAT
                bodies[it.body] = ms.worldbody.add_body(
                    name=f"b{it.body}", mocap=True, pos=list(pos), quat=list(quat)
                )
            mb = bodies[it.body]
            kw = dict(
                pos=list(it.pos), quat=list(it.quat), rgba=list(it.rgba), contype=0, conaffinity=0
            )
            if it.kind == "box":
                mb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in it.size], **kw)
            elif it.kind == "sphere":
                mb.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[it.size[0], 0, 0], **kw)
            elif it.kind == "cylinder":
                mb.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[it.size[0], it.size[1] / 2, 0], **kw
                )
            elif it.kind == "capsule":
                mb.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[it.size[0], it.size[1] / 2, 0], **kw
                )
            elif it.kind == "plane":
                h = self._extent_plane
                mb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[h, h, 0.1], **kw)
            elif it.kind == "mesh":
                n_mesh += 1
                mm = ms.add_mesh(name=f"m{n_mesh}")
                mm.uservert = it.vertices.astype(np.float32).ravel()
                mm.userface = it.faces.astype(np.int32).ravel()
                mb.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"m{n_mesh}", **kw)
        ms.worldbody.add_light(pos=[0, 0, 12], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5])
        self.model = ms.compile()
        self.data = mujoco.MjData(self.model)
        self._mocap = {
            b: int(
                self.model.body_mocapid[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"b{b}")
                ]
            )
            for b in bodies
        }
        self._gl = None

    def _plane_half_extent(self) -> float:
        pts = [(self.reg.bodies[i.body].world_pos if self.reg.bodies[i.body].static else np.zeros(3), i.radius)
               for i in self.items if i.kind != "plane"]  # fmt: skip
        reach = max([float(np.linalg.norm(p[:2])) + r for p, r in pts], default=1.0)
        return float(max(5.0, 2.0 * reach))

    def scene_bounds(self, frame):
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for it in self.items:
            if it.kind == "plane":
                continue
            info = self.reg.bodies[it.body]
            if info.static:
                p, q = info.world_pos, info.world_quat
            else:
                p, q = frame.body_pos[info.dyn], frame.body_quat[info.dyn]
            c = p + m3.quat_rotate(q, it.pos)
            r = it.radius
            lo, hi = np.minimum(lo, c - r), np.maximum(hi, c + r)
        if not np.all(np.isfinite(lo)):
            lo, hi = -np.ones(3), np.ones(3)
        return lo, hi

    def render(
        self,
        frame,
        *,
        camera: dict | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray:
        """RGB uint8 image (height, width, 3). ``camera``: dict with lookat, distance, azimuth, elevation."""
        mj = self._mj
        w, h = width or self.width, height or self.height
        for b, mid in self._mocap.items():
            info = self.reg.bodies[b]
            if not info.static:
                self.data.mocap_pos[mid] = frame.body_pos[info.dyn]
                self.data.mocap_quat[mid] = frame.body_quat[info.dyn]
        mj.mj_forward(self.model, self.data)
        lo, hi = self.scene_bounds(frame)
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        spec = {
            "lookat": (lo + hi) / 2,
            "distance": max(1.2 * float(np.linalg.norm(hi - lo)), 1.0),
            "azimuth": 135.0,
            "elevation": -25.0,
        }
        spec.update(camera or {})
        cam.lookat[:] = spec["lookat"]
        cam.distance, cam.azimuth, cam.elevation = (
            spec["distance"],
            spec["azimuth"],
            spec["elevation"],
        )
        if self._gl is None or (self._gl.width, self._gl.height) != (w, h):
            if self._gl is not None:
                self._gl.close()
            self._gl = mj.Renderer(self.model, height=h, width=w)
        self._gl.update_scene(self.data, camera=cam)
        return self._gl.render()

    def close(self) -> None:
        if self._gl is not None:
            self._gl.close()
            self._gl = None


def write_png(path, image: np.ndarray) -> Path:
    """Write an (h, w, 3|4) uint8 array as PNG (no imaging dependency)."""
    path = Path(path)
    img = np.ascontiguousarray(image, dtype=np.uint8)
    h, w, c = img.shape
    color_type = {3: 2, 4: 6}[c]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    )
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


# ---------------------------------------------------------------------------------- Rerun
def log_rerun(source, app_id: str = "lobs", *, save=None, spawn: bool = False, every: int = 1, measures: bool = True,
              anchors: bool = True) -> None:  # fmt: skip
    """Log a finished run to Rerun. ``source`` is a Simulation (its recorded frames) or a Trajectory.

    ``save="run.rrd"`` writes a recording file (open with ``rerun run.rrd``); ``spawn=True`` opens the viewer.
    """
    import rerun as rr

    from .measure.context import EvalCtx
    from .registry import build_registry
    from .trajectory import Trajectory

    if isinstance(source, Trajectory):
        spec, frames, mvals = source.spec, list(source.frames()), source.measures
        reg = build_registry(spec, source.base_dir)
        times = source.t
        fhist = source.forces
        bhist = source.behaviors
        bnames = source.meta.get("behavior_states", {})
    else:
        spec, reg = source.spec, source.reg
        frames, times = list(source._frames), source.history.time
        mvals = {n: source.history[n] for n in source.history.names()}
        fh = source._force_hist
        fhist = (
            {k: np.stack([h[k] for h in fh]) for k in fh[0]}
            if fh and len(fh) == len(frames)
            else {}
        )
        bh = source._beh_hist
        bhist = (
            {k: np.array([h[k] for h in bh]) for k in bh[0]} if bh and len(bh) == len(frames) else {}
        )
        bnames = (
            {b.spec.name: b.state_names for b in source._beh.items} if source._beh is not None else {}
        )
    if not frames:
        raise LobsError("nothing recorded to visualize (simulate with record=True)")
    rr.init(app_id, spawn=spawn)
    if save is not None:
        rr.save(str(save))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    items = drawables(spec, reg)
    for k, it in enumerate(items):
        base = f"world/{reg.bodies[it.body].path}"
        path = f"{base}/geom{k}"
        rr.log(
            path,
            rr.Transform3D(
                translation=it.pos,
                rotation=rr.Quaternion(xyzw=[it.quat[1], it.quat[2], it.quat[3], it.quat[0]]),
            ),
            static=True,
        )
        color = [int(255 * c) for c in it.rgba]
        if it.kind == "box":
            rr.log(
                path, rr.Boxes3D(half_sizes=[[s / 2 for s in it.size]], colors=[color]), static=True
            )
        elif it.kind == "sphere":
            rr.log(
                path, rr.Ellipsoids3D(half_sizes=[[it.size[0]] * 3], colors=[color]), static=True
            )
        elif it.kind == "cylinder":
            rr.log(
                path,
                rr.Cylinders3D(lengths=[it.size[1]], radii=[it.size[0]], colors=[color]),
                static=True,
            )
        elif it.kind == "capsule":
            rr.log(
                path,
                rr.Capsules3D(lengths=[it.size[1]], radii=[it.size[0]], colors=[color]),
                static=True,
            )
        elif it.kind == "plane":
            rr.log(
                path,
                rr.Boxes3D(
                    half_sizes=[[50.0, 50.0, 0.01]], centers=[[0, 0, -0.01]], colors=[color]
                ),
                static=True,
            )
        else:
            rr.log(
                path,
                rr.Mesh3D(
                    vertex_positions=it.vertices, triangle_indices=it.faces, albedo_factor=color
                ),
                static=True,
            )
    for info in reg.bodies:  # static bodies: pose once
        if info.static:
            q = info.world_quat
            rr.log(
                f"world/{info.path}",
                rr.Transform3D(
                    translation=info.world_pos,
                    rotation=rr.Quaternion(xyzw=[q[1], q[2], q[3], q[0]]),
                ),
                static=True,
            )
    for i in range(0, len(frames), max(1, every)):
        f = frames[i]
        rr.set_time("sim_time", duration=float(times[i]))
        for gi in reg.dynamic_order:
            k = reg.bodies[gi].dyn
            q = f.body_quat[k]
            rr.log(
                f"world/{reg.bodies[gi].path}",
                rr.Transform3D(
                    translation=f.body_pos[k], rotation=rr.Quaternion(xyzw=[q[1], q[2], q[3], q[0]])
                ),
            )
        if anchors and reg.anchors:
            ctx = EvalCtx(reg, f)
            for ai, a in enumerate(reg.anchors):
                from .registry import Thing

                p, rot, _ = ctx.pose(Thing("anchor", ai, path=a.path))
                rr.log(
                    f"anchors/{a.path}", rr.Transform3D(translation=p, mat3x3=rot, axis_length=0.1)
                )
        for bname, states in bhist.items():
            s = int(states[i])
            rr.log(f"behaviors/{bname}/state", rr.Scalars(float(s)))
            if i == 0 or s != int(states[max(i - every, 0)]):
                names = bnames.get(bname, [])
                label = names[s] if s < len(names) else str(s)
                rr.log(f"behaviors/{bname}/log", rr.TextLog(f"{bname} -> {label}"))
        if fhist:
            ctx = EvalCtx(reg, f)
            for fs in spec.forces:
                if fs.name not in fhist:
                    continue
                vec = fhist[fs.name][i]
                origin = ctx.pose(reg.resolve(fs.on))[0]
                scale = 0.01  # metres of arrow per newton
                rr.log(
                    f"forces/{fs.name}",
                    rr.Arrows3D(origins=[origin], vectors=[vec * scale], colors=[[230, 60, 60]]),
                )
        if measures:
            for name, arr in mvals.items():
                v = np.asarray(arr[i], dtype=float).ravel()
                if v.size == 1:
                    rr.log(f"measures/{name}", rr.Scalars(float(v[0])))
                else:
                    for j, x in enumerate(v):
                        rr.log(f"measures/{name}/{j}", rr.Scalars(float(x)))
