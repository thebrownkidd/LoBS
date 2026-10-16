"""Decision log, overrides and profile loading. Every physics decision made during import is a
`Decision` with its provenance and confidence, so imports are reviewable and reproducible."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path

import yaml

from ..errors import ImportFailure

PROVENANCE = ("override", "source", "rule", "mode", "heuristic", "default", "material-library")
CONFIDENCE = ("high", "medium", "low")


@dataclass
class Decision:
    target: str
    field: str
    value: object
    provenance: str
    confidence: str = "high"
    rationale: str = ""
    review: bool = False


@dataclass
class DecisionLog:
    items: list = field(default_factory=list)

    def add(self, target, field_, value, provenance, confidence="high", rationale="", review=False):
        assert provenance in PROVENANCE and confidence in CONFIDENCE, (provenance, confidence)
        if isinstance(value, float):
            value = float(f"{value:.9g}")
        self.items.append(
            Decision(str(target), field_, value, provenance, confidence, rationale, review)
        )

    def to_json(self) -> str:
        return json.dumps(
            [asdict(d) for d in self.items], indent=2, default=_jsonable, sort_keys=False
        )

    def counts(self) -> dict:
        out: dict = {}
        for d in self.items:
            out.setdefault(d.provenance, {}).setdefault(d.confidence, 0)
            out[d.provenance][d.confidence] += 1
        return out

    def needs_review(self) -> list:
        return [d for d in self.items if d.review]


def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


@dataclass
class ImportReport:
    source: str
    format: str
    profile: str
    motion: str | None
    counts: dict = field(default_factory=dict)  # actors / dynamic / fixed / geoms / triangles
    decisions: DecisionLog = field(default_factory=DecisionLog)
    warnings: list = field(default_factory=list)
    georeference: dict | None = None
    capabilities_required: list = field(default_factory=list)
    dropped: list = field(default_factory=list)  # features present in the source but not imported

    @property
    def needs_review(self) -> list:
        return self.decisions.needs_review()

    def markdown(self) -> str:
        lines = [f"# Import report: {Path(self.source).name}", ""]
        lines += [
            f"- format: `{self.format}`   profile: `{self.profile}`   motion mode: `{self.motion}`"
        ]
        lines += [f"- counts: {', '.join(f'{k}={v}' for k, v in self.counts.items())}"]
        if self.georeference:
            lines += [f"- georeference offset (source units): {self.georeference}"]
        if self.capabilities_required:
            lines += [
                f"- requires backend capabilities: {', '.join(sorted(self.capabilities_required))}"
            ]
        lines += ["", "## Decisions by provenance / confidence", ""]
        for prov, per in self.decisions.counts().items():
            lines.append(f"- **{prov}**: " + ", ".join(f"{c}={n}" for c, n in per.items()))
        if self.dropped:
            lines += ["", "## Not imported", ""] + [f"- {d}" for d in self.dropped]
        if self.warnings:
            lines += ["", "## Warnings", ""] + [f"- {w}" for w in self.warnings]
        review = self.needs_review
        lines += ["", f"## Needs review ({len(review)})", ""]
        for d in review[:500]:
            lines.append(
                f"- `{d.target}` {d.field} = `{d.value}` ({d.provenance}, {d.confidence}): {d.rationale}"
            )
        if len(review) > 500:
            lines.append(f"- ... and {len(review) - 500} more (see decisions.json)")
        lines += [
            "",
            "Edit `import/overrides.yaml` (keyed by source id) and run `lobs import --update` to change any of these.",
        ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- overrides
@dataclass
class Overrides:
    elements: dict = field(default_factory=dict)
    used: set = field(default_factory=set)

    @classmethod
    def load(cls, path_or_dict) -> Overrides:
        if path_or_dict is None:
            return cls()
        if isinstance(path_or_dict, dict):
            data = path_or_dict
        else:
            p = Path(path_or_dict)
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls(elements={str(k): dict(v) for k, v in (data.get("elements") or {}).items()})

    def get(self, source_id: str) -> dict:
        ov = self.elements.get(source_id, {})
        if ov:
            self.used.add(source_id)
        return ov

    def unused(self) -> list:
        return sorted(set(self.elements) - self.used)


# ---------------------------------------------------------------- profiles / materials
def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_data(name: str) -> dict:
    text = resources.files("lobs.importers.data").joinpath(name).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def load_profile(name_or_path) -> tuple[dict, str]:
    """Resolve a profile (built-in name or YAML path), following ``extends``. Returns (dict, sha256)."""
    if isinstance(name_or_path, dict):
        prof = copy.deepcopy(name_or_path)
    else:
        p = Path(str(name_or_path))
        if p.suffix in (".yaml", ".yml") and p.exists():
            prof = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        else:
            try:
                prof = _read_data(f"{name_or_path}.yaml")
            except (FileNotFoundError, ModuleNotFoundError) as e:
                raise ImportFailure(f"unknown profile '{name_or_path}'") from e
    parent = prof.pop("extends", None)
    if parent:
        base, _ = load_profile(parent)
        prof = _merge(base, prof)
    digest = hashlib.sha256(json.dumps(prof, sort_keys=True, default=str).encode()).hexdigest()
    return prof, digest


def load_materials(profile: dict) -> dict:
    mats = _read_data("materials.yaml")
    for name, m in (profile.get("materials") or {}).items():
        mats[name] = _merge(mats.get(name, {"keywords": []}), m)
    return mats
