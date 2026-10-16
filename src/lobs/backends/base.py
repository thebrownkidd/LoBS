from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ..registry import Registry, ResolvedInitial
from ..schema import WorldSpec
from ..state import StateFrame
from .capabilities import Capability, Support


class PhysicsBackend(ABC):
    """Engine adapter. Backends exchange numpy arrays and StateFrames only (no engine types)."""

    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def available(cls) -> bool: ...

    @classmethod
    @abstractmethod
    def capabilities(cls) -> dict[Capability, Support]: ...

    @classmethod
    @abstractmethod
    def engine_version(cls) -> str: ...

    @abstractmethod
    def __init__(self, spec: WorldSpec, registry: Registry, options) -> None: ...

    @abstractmethod
    def reset(self, initial: ResolvedInitial, seed: int | None) -> None: ...

    @abstractmethod
    def set_actuator_targets(self, u: np.ndarray) -> None:
        """Targets in registry actuator order; held until changed."""

    @abstractmethod
    def add_wrench(self, body: int, force_w, torque_w, point_w) -> None:
        """World-frame wrench at a world point, applied during the next ``step`` only."""

    @abstractmethod
    def step(self, substeps: int) -> None: ...

    @abstractmethod
    def read_state(self, *, contacts: bool, accelerations: bool) -> StateFrame: ...

    @abstractmethod
    def info(self) -> dict: ...

    @property
    def native(self):
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027
        pass
