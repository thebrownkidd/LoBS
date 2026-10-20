"""Build the native kernel (src/native) into src/lobs/_native*.pyd.

    pixi run python tools/build_native.py [--debug]

Needs a C++17 compiler (MSVC Build Tools on Windows: found via vcvarsall), cmake, ninja and nanobind (the pixi
default env has them). MuJoCo headers for the installed version are downloaded once into build/.
"""

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"


def mujoco_headers() -> Path:
    import mujoco

    ver = mujoco.__version__
    out = BUILD / f"mujoco-{ver}"
    if not (out / "include" / "mujoco" / "mujoco.h").exists():
        plat = {"win32": "windows-x86_64", "linux": "linux-x86_64", "darwin": "macos-universal2"}[
            sys.platform
        ]
        ext = "zip" if sys.platform == "win32" else "tar.gz"
        url = f"https://github.com/google-deepmind/mujoco/releases/download/{ver}/mujoco-{ver}-{plat}.{ext}"
        BUILD.mkdir(exist_ok=True)
        archive = BUILD / f"mujoco-{ver}.{ext}"
        print(f"downloading MuJoCo {ver} headers: {url}")
        urllib.request.urlretrieve(url, archive)
        if ext == "zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(out, [n for n in z.namelist() if n.startswith("include/")])
        else:
            import tarfile

            with tarfile.open(archive) as t:
                t.extractall(
                    out,
                    [
                        m
                        for m in t.getmembers()
                        if "/include/" in m.name or m.name.startswith("include/")
                    ],
                )
    return out / "include"


def vcvars() -> str | None:
    if sys.platform != "win32":
        return None
    roots = [
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("ProgramFiles", r"C:\Program Files"),
    ]
    for root in roots:
        for year in ("2022", "2019", "2017"):
            for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
                p = (
                    Path(root)
                    / "Microsoft Visual Studio"
                    / year
                    / edition
                    / "VC"
                    / "Auxiliary"
                    / "Build"
                    / "vcvarsall.bat"
                )
                if p.exists():
                    return str(p)
    return None


def run(cmd: str, env_prefix: str = "") -> None:
    full = f"{env_prefix}{cmd}"
    print("+", cmd)
    r = subprocess.run(full, shell=True, cwd=ROOT)
    if r.returncode:
        raise SystemExit(r.returncode)


def main() -> None:
    debug = "--debug" in sys.argv
    inc = mujoco_headers()
    build = BUILD / "native"
    py = sys.executable
    cfg = "Debug" if debug else "Release"
    prefix = ""
    vc = vcvars()
    if sys.platform == "win32":
        if vc is None:
            raise SystemExit("MSVC Build Tools not found (install 'Desktop development with C++')")
        prefix = f'call "{vc}" x64 >nul && '
    run(
        f'cmake -S src/native -B "{build}" -G Ninja -DCMAKE_BUILD_TYPE={cfg} -DPython_EXECUTABLE="{py}" -DMUJOCO_INCLUDE="{inc}"',
        prefix,
    )
    run(f'cmake --build "{build}"', prefix)
    built = sorted(build.glob("_native*.pyd")) + sorted(build.glob("_native*.so"))
    if not built:
        raise SystemExit("build finished but no _native module was produced")
    dest = ROOT / "src" / "lobs" / built[0].name
    shutil.copy2(built[0], dest)
    print(f"installed {dest}")


if __name__ == "__main__":
    main()
