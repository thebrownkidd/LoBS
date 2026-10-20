"""3D viewport: renders through the library's engine-independent MuJoCo visual twin, with Qt overlays."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..core.camera import OrbitCamera
from ..core.picking import ScenePicker

MAX_RENDER_WIDTH = 960
MAX_RENDER_HEIGHT = 1100  # below the renderer framebuffer (1200)


class Viewport(QWidget):
    selected = Signal(str)  # body path ('' = nothing)

    def __init__(self, ctl, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.cam = OrbitCamera()
        self.picker: ScenePicker | None = None
        self.sim_id = None
        self.image: QImage | None = None
        self.selection = ""
        self.show_forces = True
        self.show_labels = True
        self.show_anchors = True
        self.force_scale = 0.01  # metres of arrow per newton
        self.follow = False
        self.error = ""
        self.replay = None  # a core.replay.Replay while scrubbing a recording
        self._renderer = None
        self._replay_id = None
        self._drag = None  # ('orbit'|'pan'|'push', last pos, extra)
        self._push = None  # (body index, target point)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------ rendering
    def show_replay(self, replay) -> None:
        self.replay = replay
        self._replay_id = None
        self.refresh()

    def clear_replay(self) -> None:
        self.replay = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.sim_id = None
        self.refresh()

    def _reg_frame(self):
        """(registry, current frame) of whatever is on screen: the replay or the live simulation."""
        if self.replay is not None:
            return self.replay.reg, self.replay.frame()
        sim = self.ctl.sim
        return (sim.reg, sim.state()) if sim is not None else (None, None)

    def _sync(self) -> bool:
        if self.replay is not None:
            if id(self.replay) != self._replay_id:
                from ...viz import Renderer

                self._replay_id = id(self.replay)
                if self._renderer is not None:
                    self._renderer.close()
                self._renderer = Renderer(self.replay.spec, self.replay.reg)
                self.picker = ScenePicker(self.replay.spec, self.replay.reg)
                self.frame_all()
            self.error = ""
            return True
        sim = self.ctl.sim
        if sim is None and not self.ctl.ensure():
            self.error = self.ctl.error or "cannot build the simulation"
            self.image = None
            return False
        sim = self.ctl.sim
        if id(sim) != self.sim_id:
            self.sim_id = id(sim)
            self.picker = ScenePicker(sim.spec, sim.reg)
            self.frame_all()
        self.error = ""
        return True

    def frame_all(self) -> None:
        reg, frame = self._reg_frame()
        if reg is None or self.picker is None:
            return
        centers, radii, _ = self.picker.spheres(frame)
        if centers:
            c, r = np.array(centers), np.array(radii)[:, None]
            self.cam.frame((c - r).min(axis=0), (c + r).max(axis=0))

    def refresh(self) -> None:
        if not self._sync():
            self.update()
            return
        reg, frame = self._reg_frame()
        if self.follow and self.selection:
            try:
                self.cam.lookat = self.picker.body_position(frame, self._body_index(self.selection))
            except Exception:  # noqa: BLE001 - selection may no longer exist after an edit
                pass
        w = min(self.width(), MAX_RENDER_WIDTH)
        h = max(int(w * self.height() / max(self.width(), 1)), 2)
        if h > MAX_RENDER_HEIGHT:
            h, w = (
                MAX_RENDER_HEIGHT,
                max(int(MAX_RENDER_HEIGHT * self.width() / max(self.height(), 1)), 2),
            )
        if self.replay is not None:
            img = self._renderer.render(frame, camera=self.cam.as_dict(), width=w, height=h)
        else:
            img = self.ctl.sim.render(w, h, camera=self.cam.as_dict())
        self.image = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.update()

    def grab_image(self) -> np.ndarray | None:
        """The last rendered frame as an RGB array (for screenshots / GIF export)."""
        if self.image is None:
            return None
        img = self.image.convertToFormat(QImage.Format.Format_RGB888)
        arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(
            img.height(), img.bytesPerLine()
        )
        return arr[:, : img.width() * 3].reshape(img.height(), img.width(), 3).copy()

    # ------------------------------------------------------------ painting
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(24, 26, 30))
        if self.image is None:
            p.setPen(QColor(200, 120, 120) if self.error else QColor(150, 150, 150))
            p.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, self.error or "Open a world to begin"
            )
            return
        p.drawImage(self.rect(), self.image)
        reg, frame = self._reg_frame()
        if reg is not None:
            self._overlays(p, reg, frame)

    def _project(self, pts):
        pix, z = self.cam.project(pts, self.width(), self.height())
        return pix, z

    def _forces(self):
        """``[(name, on_path, vector)]`` of the active force laws now (live or recorded)."""
        if self.replay is not None:
            on = {f.name: f.on for f in self.replay.spec.forces}
            return [
                (k, on[k], v)
                for k, v in self.replay.forces_at().items()
                if np.linalg.norm(v) > 1e-9 and k in on
            ]
        sim = self.ctl.sim
        return [
            (law.name, law.on.path, law.last_force)
            for law in sim._laws
            if law.active and np.linalg.norm(law.last_force) > 1e-9
        ]

    def _labels(self):
        """``[(actor_path, text)]`` behavior labels now."""
        if self.replay is not None:
            return self.replay.labels_at()
        sim = self.ctl.sim
        if sim is None or sim._beh is None:
            return []
        return [
            (b.on.path, f"{b.spec.name}: {b.state_name}" if b.machine is not None else b.spec.name)
            for b in sim._beh.items
            if b.on is not None
        ]

    def _overlays(self, p: QPainter, reg, frame) -> None:
        from ...measure.context import EvalCtx
        from ...registry import Thing

        font = QFont()
        font.setPointSize(8)
        p.setFont(font)
        ctx = EvalCtx(reg, frame)
        if self.show_anchors and reg.anchors:
            pts = np.array(
                [ctx.pose(Thing("anchor", i, path=a.path))[0] for i, a in enumerate(reg.anchors)]
            )
            pix, z = self._project(pts)
            p.setPen(QPen(QColor(255, 210, 90), 2))
            for (x, y), zz in zip(pix, z, strict=True):
                if zz > 0:
                    p.drawEllipse(QPointF(x, y), 3, 3)
        if self.selection:
            try:
                centre = self.picker.body_position(frame, self._body_index(self.selection))
                pix, z = self._project(centre)
                if z[0] > 0:
                    p.setPen(QPen(QColor(90, 200, 255), 2))
                    p.drawEllipse(QPointF(*pix[0]), 14, 14)
                    p.drawText(QPointF(pix[0, 0] + 16, pix[0, 1] - 6), self.selection)
            except Exception:  # noqa: BLE001
                pass
        if self.show_forces:
            p.setPen(QPen(QColor(235, 70, 70), 3))
            for name, on, f in self._forces():
                o = ctx.pose(reg.resolve(on))[0]
                pix, z = self._project(np.array([o, o + np.asarray(f) * self.force_scale]))
                if z.min() > 0:
                    p.drawLine(QPointF(*pix[0]), QPointF(*pix[1]))
                    p.drawEllipse(QPointF(*pix[1]), 3, 3)
                    p.drawText(QPointF(pix[1, 0] + 6, pix[1, 1]), name)
        if self.show_labels:
            p.setPen(QColor(255, 255, 255))
            for actor, text in self._labels():
                pix, z = self._project(ctx.pose(reg.resolve(actor))[0] + np.array([0, 0, 1.0]))
                if z[0] > 0:
                    p.drawText(QPointF(pix[0, 0] - 30, pix[0, 1]), text)
        p.setPen(QColor(220, 220, 220))
        if self.replay is not None:
            p.drawText(
                8,
                16,
                f"REPLAY  t = {frame.t:7.3f} s   frame {self.replay.index + 1}/{len(self.replay)}",
            )
        else:
            sim = self.ctl.sim
            p.drawText(8, 16, f"t = {sim.time:7.3f} s   tick {sim.tick}   {self.ctl.backend}")

    # ------------------------------------------------------------ interaction
    def _body_index(self, path: str) -> int:
        reg = self._reg_frame()[0]
        return (
            reg.resolve(path).index
            if reg.resolve(path).kind == "body"
            else reg.actors[reg.resolve(path).index].bodies[0]
        )

    def mousePressEvent(self, e):
        self.setFocus()
        pos = e.position()
        mods = e.modifiers()
        if e.button() == Qt.MouseButton.MiddleButton or (
            e.button() == Qt.MouseButton.LeftButton and mods & Qt.KeyboardModifier.ShiftModifier
        ):
            self._drag = ("pan", pos)
        elif (
            e.button() == Qt.MouseButton.LeftButton
            and self.picker is not None
            and self._reg_frame()[0] is not None
        ):
            reg, frame = self._reg_frame()
            hit = self.picker.pick(self.cam, pos.x(), pos.y(), self.width(), self.height(), frame)
            can_push = self.replay is None and hit is not None and not reg.bodies[hit].static
            if mods & Qt.KeyboardModifier.ControlModifier and can_push:
                self._push = (hit, self.picker.body_position(frame, hit))
                self._drag = ("push", pos)
            else:
                self._drag = ("orbit", pos, hit)
        elif e.button() == Qt.MouseButton.RightButton:
            self._drag = ("pan", pos)

    def mouseMoveEvent(self, e):
        if not self._drag:
            return
        kind, last = self._drag[0], self._drag[1]
        pos = e.position()
        dx, dy = pos.x() - last.x(), pos.y() - last.y()
        if kind == "orbit":
            self.cam.orbit(dx, dy)
        elif kind == "pan":
            self.cam.pan(dx, dy, self.height())
        elif kind == "push" and self._push is not None:
            o, d = self.cam.ray(pos.x(), pos.y(), self.width(), self.height())
            body, anchor = self._push
            depth = float(np.dot(anchor - o, d))
            self._push = (body, o + d * depth)
        self._drag = (kind, pos, *self._drag[2:])
        if not self.ctl.playing:
            self.refresh()
        else:
            self.update()

    def mouseReleaseEvent(self, e):
        d, self._drag, self._push = self._drag, None, None
        if d and d[0] == "orbit" and len(d) > 2:
            moved = abs(e.position().x() - d[1].x()) + abs(e.position().y() - d[1].y())
            if moved < 4:  # a click, not a drag
                hit = d[2]
                self.selection = "" if hit is None else self.picker.body_path(hit)
                self.selected.emit(self.selection)
                self.update()

    def wheelEvent(self, e):
        self.cam.zoom(e.angleDelta().y() / 120.0)
        if not self.ctl.playing:
            self.refresh()

    def apply_push(self) -> None:
        """Spring-damper toward the cursor target while Ctrl-dragging a body (called every frame)."""
        if self._push is None or self.ctl.sim is None:
            return
        body, target = self._push
        sim = self.ctl.sim
        info = sim.reg.bodies[body]
        frame = sim.state()
        p, v = frame.body_pos[info.dyn], frame.body_linvel[info.dyn]
        m = info.mass
        f = m * (30.0 * (target - p) - 8.0 * v)
        sim.apply_force(info.path, f, ticks=1)
