"""Print engine versions, run a trivial step in each, and dump key engine APIs to docs/engine_api/."""

import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "engine_api"


def dump(name: str, objs: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for label, obj in objs.items():
        lines.append(f"== {label} ==")
        lines.extend(n for n in dir(obj) if not n.startswith("__"))
        lines.append("")
    (OUT / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")


def probe_mujoco() -> None:
    import mujoco

    print("mujoco", mujoco.__version__)
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(pos=[0, 0, 1])
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1], mass=1.0)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    print("  stepped; z =", data.xpos[1][2])
    dump(
        "mujoco_3",
        {
            "mujoco": mujoco,
            "MjSpec": mujoco.MjSpec,
            "MjsBody": type(body),
            "MjsGeom": type(body.add_geom(size=[0.1, 0.1, 0.1])),
            "MjsJoint": type(body.add_joint()),
            "MjsActuator": type(spec.add_actuator()),
        },
    )


def probe_chrono() -> None:
    import pychrono as chrono

    # Chrono 10 has no runtime version attribute; read it from conda metadata.
    meta = sorted((Path(sys.prefix) / "conda-meta").glob("pychrono-*.json"))
    print("pychrono", meta[0].stem if meta else "unknown (no conda-meta)")
    import pychrono.fea

    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    body = chrono.ChBodyAuxRef()
    system.Add(body)
    system.DoStepDynamics(0.002)
    print("  stepped; t =", system.GetChTime())
    dump(
        "chrono_10",
        {
            "pychrono": chrono,
            "ChSystemNSC": chrono.ChSystemNSC,
            "ChBodyAuxRef": chrono.ChBodyAuxRef,
            "ChLinkLockRevolute": chrono.ChLinkLockRevolute,
            "ChLinkLockPrismatic": chrono.ChLinkLockPrismatic,
            "ChContactMaterialNSC": chrono.ChContactMaterialNSC,
        },
    )


if __name__ == "__main__":
    for fn in (probe_mujoco, probe_chrono):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - probes must report, not crash
            print(f"{fn.__name__} FAILED: {type(e).__name__}: {e}")
