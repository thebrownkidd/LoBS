"""Render a baked episode (anim.json) to video with the Three.js player.

The player advances only when asked for a frame, so capture is deterministic: same anim.json, same pixels.

    pixi run -e lob lob-video --anim runs/lobpass/footage/ep12/anim.json --out runs/lobpass/footage/ep12/ep12.mp4
    pixi run -e lob lob-video --anim ... --stills 0 120 240 --png-dir /tmp/stills        # a few frames as PNG
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import functools
import http.server
import pathlib
import subprocess
import threading

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHROME_ARGS = [
    "--use-angle=d3d11",
    "--ignore-gpu-blocklist",
    "--enable-webgl",
    "--enable-unsafe-swiftshader",
]


@contextlib.contextmanager
def serve(root: pathlib.Path = ROOT):
    handler = functools.partial(_Quiet, directory=str(root))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):  # silence request logging
        pass


def _rel(path: pathlib.Path) -> str:
    return "/" + pathlib.Path(path).resolve().relative_to(ROOT).as_posix()


@contextlib.contextmanager
def open_player(anim: pathlib.Path, hud: bool = True):
    from playwright.sync_api import sync_playwright

    with serve() as port, sync_playwright() as pw:
        browser = pw.chromium.launch(args=CHROME_ARGS)
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        errors: list[str] = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        url = (
            f"http://127.0.0.1:{port}/demos/lobpass/player/index.html?anim={_rel(anim)}&capture=1"
            + ("" if hud else "&nohud=1")
        )
        page.goto(url)
        page.wait_for_function(
            "window.lobpass && window.lobpass.ready || window.lobpassError", timeout=120000
        )
        err = page.evaluate("window.lobpassError || null")
        if err:
            browser.close()
            raise RuntimeError(f"player failed: {err}; console: {errors[:5]}")
        try:
            yield page, errors
        finally:
            browser.close()


def grab(page, n: int, hud: bool = True) -> bytes:
    page.evaluate("n => window.lobpass.setFrame(n)", n)
    if hud:  # a page screenshot includes the HTML HUD; CDP with optimizeForSpeed is ~3x faster than page.screenshot
        cdp = getattr(page, "_lobpass_cdp", None)
        if cdp is None:
            cdp = page._lobpass_cdp = page.context.new_cdp_session(page)
        return base64.b64decode(
            cdp.send("Page.captureScreenshot", {"format": "png", "optimizeForSpeed": True})["data"]
        )
    url = page.evaluate("window.lobpass.grab('image/png')")
    return base64.b64decode(url.split(",", 1)[1])


def render_stills(anim, frames, out_dir, hud=True):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with open_player(pathlib.Path(anim), hud) as (page, errors):
        for n in frames:
            p = out_dir / f"frame_{n:05d}.png"
            p.write_bytes(grab(page, n, hud))
            paths.append(p)
    return paths, errors


def render_video(
    anim, out, frames: tuple | None = None, hud: bool = True, crf: int = 16, quiet: bool = False
):
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open_player(pathlib.Path(anim), hud) as (page, errors):
        total = page.evaluate("window.lobpass.frames")
        a, b = (0, total) if frames is None else frames
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", "60", "-i", "-",
               "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-profile:v", "high", "-pix_fmt", "yuv420p",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-r", "60", str(out)]  # fmt: skip
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for n in range(a, b):
            ff.stdin.write(grab(page, n, hud))
            if not quiet and (n - a) % 60 == 0:
                print(f"  frame {n - a}/{b - a}", flush=True)
        ff.stdin.close()
        ff.wait()
    return out, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anim", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stills", type=int, nargs="*", default=None)
    ap.add_argument("--png-dir", default="stills")
    ap.add_argument("--nohud", action="store_true")
    args = ap.parse_args()
    if args.stills is not None:
        paths, errors = render_stills(args.anim, args.stills, args.png_dir, not args.nohud)
        print("\n".join(str(p) for p in paths), "\nconsole errors:", errors)
    else:
        out, errors = render_video(
            args.anim, args.out or pathlib.Path(args.anim).with_suffix(".mp4"), hud=not args.nohud
        )
        print(f"{out} (console errors: {errors})")


if __name__ == "__main__":
    main()
