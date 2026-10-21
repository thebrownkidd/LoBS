"""An animated "training dashboard" clip from real evaluation numbers: the learning curve of the policy and, for a
handful of checkpoints, what its plays look like (outcome shares). Rendered with matplotlib straight to MP4.

    pixi run -e lob python -m demos.lobpass.dashboard --rows runs/lobpass/footage/dashboard_rows.json --out runs/lobpass/footage/dashboard.mp4

``rows`` is a list of {"label", "decisions" (millions, x position), "success", "outcomes": {...}, "kind": "rl"|"clone"|"final"}.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

ORDER = ["DUNK", "CATCH_NO_DUNK", "MISS", "INTERCEPTED", "BLOCKED", "NO_PASS"]
COLORS = {"DUNK": "#27c46b", "CATCH_NO_DUNK": "#9bd9b4", "MISS": "#8b93a1", "INTERCEPTED": "#e5484d", "BLOCKED": "#f08a4b", "NO_PASS": "#4b5160"}
BG, FG, MUTE, ACCENT = "#0e1117", "#ffffff", "#8b93a1", "#7aa7ff"


def _ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def render(rows: list[dict], out: pathlib.Path, title: str = "Training", subtitle: str = "", seconds: float = 10.0, fps: int = 60) -> pathlib.Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": MUTE, "xtick.color": MUTE, "ytick.color": MUTE})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    fig.text(0.05, 0.93, title, fontsize=44, fontweight="bold", color=FG)
    fig.text(0.05, 0.885, subtitle, fontsize=20, color=MUTE)
    ax = fig.add_axes([0.07, 0.12, 0.50, 0.68], facecolor=BG)
    bx = fig.add_axes([0.66, 0.19, 0.30, 0.61], facecolor=BG)
    rl = [r for r in rows if r.get("kind") == "rl"]
    xs = np.array([r["decisions"] for r in rl])
    ys = np.array([r["success"] for r in rl]) * 100
    others = [r for r in rows if r.get("kind") != "rl"]
    xmax = max([r["decisions"] for r in rows] + [1.0]) * 1.05
    n_frames = int(seconds * fps)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1,
                          extra_args=["-crf", "16", "-pix_fmt", "yuv420p", "-preset", "medium"])  # fmt: skip
    with writer.saving(fig, str(out), dpi=100):
        for f in range(n_frames):
            u = f / max(n_frames - 1, 1)
            ax.clear()
            bx.clear()
            for a in (ax, bx):
                a.set_facecolor(BG)
                for s in a.spines.values():
                    s.set_color("#2a3040")
            ax.set_xlim(0, xmax)
            ax.set_ylim(0, 100)
            ax.set_xlabel("training decisions (millions)", fontsize=16)
            ax.set_ylabel("plays that end in a dunk (%)", fontsize=16)
            ax.grid(color="#1c2230", linewidth=1)
            # the learning curve of reinforcement learning from scratch, drawn progressively
            if len(xs):
                p = _ease(u / 0.55)
                k = p * (len(xs) - 1)
                i = int(np.floor(k))
                cx = list(xs[: i + 1])
                cy = list(ys[: i + 1])
                if i < len(xs) - 1:
                    t = k - i
                    cx.append(xs[i] + t * (xs[i + 1] - xs[i]))
                    cy.append(ys[i] + t * (ys[i + 1] - ys[i]))
                ax.plot(cx, cy, color=ACCENT, linewidth=5, solid_capstyle="round", label="RL from scratch")
                ax.scatter(cx[:-1] if i < len(xs) - 1 else cx, cy[:-1] if i < len(xs) - 1 else cy, s=120, color=ACCENT, zorder=3)
                if u >= 0.55:
                    ax.annotate(f"{ys[-1]:.0f}%", (xs[-1], ys[-1]), textcoords="offset points", xytext=(12, -6), fontsize=22, color=ACCENT, fontweight="bold")
            # imitation of the expert, and the final policy: horizontal markers that arrive later
            for j, r in enumerate(others):
                a0 = 0.55 + 0.4 * j / max(len(others), 1)
                a = _ease((u - a0) / 0.12)
                if a > 0:
                    col = "#27c46b" if r.get("kind") == "final" else "#ff9a3c"
                    ax.scatter([r["decisions"]], [r["success"] * 100], s=420 * a, color=col, zorder=4, marker="*" if r.get("kind") == "final" else "D")
                    ax.annotate(f"{r['label']}  {r['success'] * 100:.0f}%", (r["decisions"], r["success"] * 100), textcoords="offset points", xytext=(16, -8 - 26 * j), fontsize=18, color=col, ha="left", alpha=a)
            if len(xs):
                ax.legend(loc="lower right", facecolor=BG, edgecolor="#2a3040", fontsize=14, labelcolor=FG)
            # outcome shares of the checkpoints, each bar growing in turn
            names = [r["label"] for r in rows]
            for j, r in enumerate(rows):
                a = _ease((u - 0.05 - 0.8 * j / len(rows)) / 0.1)
                bottom = 0.0
                for k in ORDER:
                    v = r["outcomes"].get(k, 0.0) * 100 * a
                    bx.barh(len(rows) - 1 - j, v, left=bottom, color=COLORS[k], height=0.7)
                    bottom += v
            bx.set_yticks(range(len(rows)))
            bx.set_yticklabels(list(reversed(names)), fontsize=13)
            bx.set_xlim(0, 100)
            bx.set_xlabel("share of plays (%)", fontsize=16)
            bx.set_title("What the plays look like", fontsize=20, color=FG, loc="left")
            handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ORDER]
            bx.legend(handles, [k.replace("_", " ").title() for k in ORDER], loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=11, facecolor=BG, edgecolor="#2a3040", labelcolor=FG)
            writer.grab_frame()
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Training")
    ap.add_argument("--subtitle", default="Stage S0: a passer, a runner, an open court")
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()
    rows = json.loads(pathlib.Path(args.rows).read_text())
    print(render(rows, pathlib.Path(args.out), args.title, args.subtitle, args.seconds))


if __name__ == "__main__":
    main()
