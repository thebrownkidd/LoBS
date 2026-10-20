"""3D viewport: renders through the library's engine-independent MuJoCo visual twin, with Qt overlays."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..core.camera import OrbitCamera
from ..core.picking import ScenePicker

MAX_RENDER_WIDTH = 960


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
        self._drag = None  # ('orbit'|'pan'|'push', last pos, extra)
        self._push = None  # (body index, target point)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------ rendering
    def _sync(self) -> bool:
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
        sim = self.ctl.sim
        if sim is None or self.picker is None:
            return
        centers, radii, _ = self.picker.spheres(sim.state())
        if centers:
            c, r = np.array(centers), np.array(radii)[:, None]
            self.cam.frame((c - r).min(axis=0), (c + r).max(axis=0))

    def refresh(self) -> None:
        if not self._sync():
            self.update()
            return
        sim = self.ctl.sim
        if self.follow and self.selection:
            try:
                self.cam.lookat = self.picker.body_position(
                    sim.state(), self._body_index(self.selection)
                )
            except Exception:  # noqa: BLE001 - selection may no longer exist after an edit
                pass
        w = min(self.width(), MAX_RENDER_WIDTH)
        h = max(int(w * self.height() / max(self.width(), 1)), 2)
        img = sim.render(w, h, camera=self.cam.as_dict())
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
        sim = self.ctl.sim
        if sim is not None:
            self._overlays(p, sim)

    def _project(self, pts):
        pix, z = self.cam.project(pts, self.width(), self.height())
        return pix, z

    def _overlays(self, p: QPainter, sim) -> None:
        frame = sim.state()
        font = QFont()
        font.setPointSize(8)
        p.setFont(font)
        if self.show_anchors and sim.reg.anchors:
            from ...measure.context import EvalCtx
            from ...registry import Thing

            ctx = EvalCtx(sim.reg, frame)
            pts = np.array(
                [
                    ctx.pose(Thing("anchor", i, path=a.path))[0]
                    for i, a in enumerate(sim.reg.anchors)
                ]
            )
            pix, z = self._project(pts)
            p.setPen(QPen(QColor(255, 210, 90), 2))
            for (x, y), zz in zip(pix, z, strict=True):
                if zz > 0:
                    p.drawEllipse(QPointF(x, y), 3, 3)
        if self.selection:
            try:
                b = self._body_index(self.selection)
                centre = self.picker.body_position(frame, b)
                pix, z = self._project(centre)
                if z[0] > 0:
                    p.setPen(QPen(QColor(90, 200, 255), 2))
                    p.drawEllipse(QPointF(*pix[0]), 14, 14)
                    p.drawText(QPointF(pix[0, 0] + 16, pix[0, 1] - 6), self.selection)
            except Exception:  # noqa: BLE001
                pass
        if self.show_forces and sim._laws:
            from ...measure.context import EvalCtx

            ctx = EvalCtx(sim.reg, frame)
            p.setPen(QPen(QColor(235, 70, 70), 3))
            for law in sim._laws:
                f = law.last_force
                if not law.active or np.linalg.norm(f) < 1e-9:
                    continue
                o = ctx.pose(law.on)[0]
                pix, z = self._project(np.array([o, o + f * self.force_scale]))
                if z.min() > 0:
                    p.drawLine(QPointF(*pix[0]), QPointF(*pix[1]))
                    p.drawEllipse(QPointF(*pix[1]), 3, 3)
                    p.drawText(QPointF(pix[1, 0] + 6, pix[1, 1]), law.name)
        if self.show_labels and sim._beh is not None:
            from ...measure.context import EvalCtx

            ctx = EvalCtx(sim.reg, frame)
            p.setPen(QColor(255, 255, 255))
            for b in sim._beh.items:
                if b.on is None:
                    continue
                top = ctx.pose(b.on)[0] + np.array([0, 0, 1.0])
                pix, z = self._project(top)
                if z[0] > 0:
                    label = (
                        f"{b.spec.name}: {b.state_name}" if b.machine is not None else b.spec.name
                    )
                    p.drawText(QPointF(pix[0, 0] - 30, pix[0, 1]), label)
        p.setPen(QColor(220, 220, 220))
        p.drawText(8, 16, f"t = {sim.time:7.3f} s   tick {sim.tick}   {self.ctl.backend}")

    # ------------------------------------------------------------ interaction
    def _body_index(self, path: str) -> int:
        reg = self.ctl.sim.reg
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
            and self.ctl.sim is not None
            and self.picker is not None
        ):
            hit = self.picker.pick(
                self.cam, pos.x(), pos.y(), self.width(), self.height(), self.ctl.sim.state()
            )
            if (
                mods & Qt.KeyboardModifier.ControlModifier
                and hit is not None
                and not self.ctl.sim.reg.bodies[hit].static
            ):
                self._push = (hit, self.picker.body_position(self.ctl.sim.state(), hit))
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
