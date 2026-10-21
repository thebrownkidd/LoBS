"""Build the footage: a learning montage (the same play by an untrained, a partly trained and the final passer), one
highlight per curriculum stage, a hero play against five defenders, and title / end cards, all rendered with the
Three.js player and joined with ffmpeg.

    pixi run -e lob python -m demos.lobpass.reel --final runs/lobpass/ft_S3c/ckpt_S3c_final.pt --out runs/lobpass/footage/reel
    pixi run -e lob python -m demos.lobpass.reel --final ... --only training      # just the learning montage
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import numpy as np

from . import constants as C
from .scenario import sample_scenario

FFMPEG = "ffmpeg"
STAGE_TITLES = {
    "S0": "Stage S0 · fixed heights, open court",
    "S1": "Stage S1 · random heights, open court",
    "S2": "Stage S2 · stationary defenders",
    "S3a": "Stage S3a · defenders that read the ball and jump",
    "S3b": "Stage S3b · defenders that chase and cut off",
    "S3c": "Stage S3c · five fast defenders",
}
_CODEC = ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high", "-pix_fmt", "yuv420p",
          "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-r", "60"]  # fmt: skip


def run_ffmpeg(args: list[str]) -> None:
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args], check=True)


# ------------------------------------------------------------------ cards
def render_card(out_png: pathlib.Path, **params) -> pathlib.Path:
    from urllib.parse import urlencode

    from playwright.sync_api import sync_playwright

    from . import capture

    out_png.parent.mkdir(parents=True, exist_ok=True)
    with capture.serve() as port, sync_playwright() as pw:
        browser = pw.chromium.launch(args=capture.CHROME_ARGS)
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page.goto(f"http://127.0.0.1:{port}/demos/lobpass/player/card.html?{urlencode(params)}")
        page.wait_for_function("window.cardReady === true", timeout=30000)
        page.screenshot(path=str(out_png))
        browser.close()
    return out_png


def card_clip(out_mp4: pathlib.Path, seconds: float = 2.5, **params) -> pathlib.Path:
    png = render_card(out_mp4.with_suffix(".png"), **params)
    fade = 0.4
    run_ffmpeg(["-loop", "1", "-framerate", "60", "-t", str(seconds), "-i", str(png), "-vf",
                f"fade=t=in:st=0:d={fade},fade=t=out:st={seconds - fade}:d={fade},format=yuv420p", *_CODEC, str(out_mp4)])  # fmt: skip
    return out_mp4


# ------------------------------------------------------------------ episodes
def episode_clip(
    name: str,
    stage: str,
    seed: int,
    checkpoint: str,
    out_dir: pathlib.Path,
    captions=None,
    sample: bool = False,
    torch_seed: int = 0,
    hud: bool = True,
    force: bool = False,
):
    """Record, bake and render one play. Returns (mp4 path, meta)."""
    from . import capture
    from .anim import bake, camera
    from .record import record_episode, save_episode

    out_dir = pathlib.Path(out_dir)
    mp4 = out_dir / f"{name}.mp4"
    ep = out_dir / name
    if mp4.exists() and (ep / "episode.json").exists() and not force:
        return mp4, json.loads((ep / "episode.json").read_text())
    arrays, meta = record_episode(
        sample_scenario(stage, seed), checkpoint, post_seconds=2.0, sample=sample, seed=torch_seed
    )
    if captions:
        rel = next((e for e in meta["events"] if e["type"] == "release"), None)
        t_rel = None if rel is None else rel["tick"] / 100.0
        meta["captions"] = captions(meta, t_rel) if callable(captions) else captions
    save_episode(arrays, meta, ep)
    baked = bake.bake_episode(arrays, meta)
    direction = camera.direct(baked)
    anim = bake.export(baked, direction, ep / "anim.json")
    capture.render_video(anim, mp4, hud=hud, quiet=True)
    return mp4, meta


def find_plays(
    checkpoint: str, stage: str, want: int, start: int, predicate=None, limit: int = 400
):
    """Seeds (>= ``start``) where the deterministic policy dunks, optionally filtered by ``predicate(info)``."""
    import torch

    from .env import LobPassEnv
    from .train import load_policy

    policy = load_policy(checkpoint)
    env = LobPassEnv({stage: 1.0})
    found = []
    for seed in range(start, start + limit):
        obs, _ = env.reset(options={"stage": stage, "scenario_seed": seed})
        while True:
            with torch.no_grad():
                a, *_ = policy.act(
                    torch.as_tensor(obs, dtype=torch.float32)[None], deterministic=True
                )
            obs, _, done, _, info = env.step(a[0].numpy())
            if done:
                break
        if info["outcome"] == "DUNK" and (predicate is None or predicate(info)):
            found.append(seed)
            if len(found) >= want:
                break
    return found


def concat(clips: list[pathlib.Path], out: pathlib.Path) -> pathlib.Path:
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{pathlib.Path(c).resolve().as_posix()}'\n" for c in clips))
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
    return out


def contact_sheet(video: pathlib.Path, out_png: pathlib.Path, n: int = 12) -> pathlib.Path:
    dur = float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(video),
            ]
        )
    )
    run_ffmpeg(
        [
            "-i",
            str(video),
            "-vf",
            f"fps={n / dur:.5f},scale=640:-1,tile=4x{int(np.ceil(n / 4))}",
            "-frames:v",
            "1",
            str(out_png),
        ]
    )
    return out_png


# ------------------------------------------------------------------ the reel
def build(
    final: str,
    out: pathlib.Path,
    only: str | None = None,
    metrics: pathlib.Path | None = None,
    early: dict | None = None,
    stage_ckpts: dict | None = None,
) -> dict:
    out = pathlib.Path(out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    clips_dir = out / "clips"
    early = early or {}
    parts: dict[str, list[pathlib.Path]] = {"training": [], "post": []}

    title = card_clip(clips_dir / "title.mp4", 2.6, kicker="LoBS", title="Learning the lob",
                      sub="A passer learns to throw an alley-oop to a runner through five defenders.")  # fmt: skip
    # ---- training footage: the same play, from an untrained passer to the final one
    if only in (None, "training"):
        seed = find_plays(final, "S1", 1, 5200, predicate=lambda i: i.get("finish") == "air")[0]
        steps = [("m0_untrained", early.get("init"), True, "Untrained: it throws at random"),
                 ("m1_rl_early", early.get("rl_early"), True, "Reinforcement learning, first 0.25 M decisions"),
                 ("m2_rl_late", early.get("rl_late"), False, "Reinforcement learning, 1.25 M decisions"),
                 ("m3_final", final, False, "Curriculum-trained: imitation of the expert, then RL refinement")]  # fmt: skip
        for name, ck, sample, text in steps:
            if not ck:
                continue
            mp4, meta = episode_clip(
                name,
                "S1",
                seed,
                ck,
                clips_dir,
                captions=[{"t0": 0.0, "t1": 99.0, "text": text}],
                sample=sample,
            )
            parts["training"].append(mp4)
        parts["training"].insert(0, card_clip(clips_dir / "training_card.mp4", 2.2, kicker="Training", title="Same play, more experience",
                                              sub=f"Scenario seed {seed}, stage S1"))  # fmt: skip
    # ---- post-training footage: one highlight per stage and a hero play
    if only in (None, "post"):
        parts["post"].append(card_clip(clips_dir / "post_card.mp4", 2.2, kicker="After training", title="One policy, six stages",
                                       sub="Open court to five fast defenders."))  # fmt: skip
        for st in C.STAGES:
            ck = (stage_ckpts or {}).get(st, final)
            pred = (lambda i: i.get("n_def", 0) >= 3) if st in ("S3b", "S3c") else None
            seeds = find_plays(ck, st, 1, 7100 + 100 * C.STAGES.index(st), predicate=pred)
            if not seeds:
                print(f"no dunk found for {st}; skipping its highlight", flush=True)
                continue
            mp4, _ = episode_clip(
                f"stage_{st}",
                st,
                seeds[0],
                ck,
                clips_dir,
                captions=[{"t0": 0.0, "t1": 99.0, "text": STAGE_TITLES[st]}],
            )
            parts["post"].append(mp4)
        hero = find_plays(final, "S3c", 1, 8100, predicate=lambda i: i.get("n_def", 0) >= 4)
        if hero:

            def caps(meta, t_rel):
                t0 = 0.0 if t_rel is None else max(t_rel - 1.4, 0.0)
                return [{"t0": 0.0, "t1": t0 + 0.6, "text": "Five defenders. One runner. One chance."},
                        {"t0": t0 + 0.6, "t1": (t_rel or 3.0) + 0.05, "text": "The passer sees the runner's plan and every defender"},
                        {"t0": (t_rel or 3.0) + 0.05, "t1": 99.0, "text": "Release"}]  # fmt: skip

            mp4, _ = episode_clip("hero", "S3c", hero[0], final, clips_dir, captions=caps)
            parts["post"].append(mp4)
        stats = ""
        if metrics and pathlib.Path(metrics).exists():
            res = {r["stage"]: r for r in json.loads(pathlib.Path(metrics).read_text())["stages"]}
            stats = "|".join(
                f"{res[s]['success'] * 100:.0f}%~{s} dunk rate"
                for s in ("S0", "S2", "S3c")
                if s in res
            )
        parts["post"].append(card_clip(clips_dir / "end.mp4", 3.2, kicker="LoBS", title="Learned, not scripted",
                                       sub="Imitation of an oracle, refined by reinforcement learning in the LoBS physics world.", stats=stats))  # fmt: skip
    outputs = {}
    for name, key in (("training_footage.mp4", "training"), ("post_training_footage.mp4", "post")):
        if parts[key]:
            seq = ([title] if key == "training" else []) + parts[key]
            outputs[key] = concat(seq, out / name)
    if parts["training"] and parts["post"]:
        outputs["reel"] = concat(
            [title, *parts["training"], *parts["post"]], out / "lobpass_reel.mp4"
        )
        outputs["sheet"] = contact_sheet(outputs["reel"], out / "contact_sheet.png")
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=["training", "post"], default=None)
    ap.add_argument(
        "--metrics", default=None, help="metrics.json from evaluate.py for the end card"
    )
    ap.add_argument("--init-ckpt", default="runs/lobpass/main/ckpt_init.pt")
    ap.add_argument("--rl-early", default="runs/lobpass/main/ckpt_S0_250k.pt")
    ap.add_argument("--rl-late", default="runs/lobpass/main/ckpt_S0_1254k.pt")
    args = ap.parse_args()
    outs = build(args.final, pathlib.Path(args.out), args.only, args.metrics,
                 early={"init": args.init_ckpt, "rl_early": args.rl_early, "rl_late": args.rl_late})  # fmt: skip
    for k, v in outs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
