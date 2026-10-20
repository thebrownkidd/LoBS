"""Write the demo world used by `lobs gui --demo` to examples/worlds/defence.yaml."""

from pathlib import Path

from lobs.gui.demo import build_demo

out = Path(__file__).with_name("defence.yaml")
build_demo().save(out)
print(f"wrote {out}")
