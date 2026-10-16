class LobsError(Exception):
    """Base class for all LoBS errors."""


class WorldValidationError(LobsError):
    def __init__(self, report):
        self.report = report
        super().__init__("world validation failed:\n" + report.format())


class BackendUnavailable(LobsError):
    pass


class CapabilityError(LobsError):
    pass


class MeasureError(LobsError):
    """A measure definition is invalid (unknown path, unit mismatch, illegal rate, ...)."""


class ImporterUnavailable(LobsError):
    """A world importer needs an optional dependency that is not installed."""


class ImportFailure(LobsError):
    """The source file could not be imported (unsupported feature, malformed file, ...)."""
