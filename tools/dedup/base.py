"""The only contract a duplicate-detection back-end has to fulfil."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

# progress(phase, done, total) - called often, must be cheap.
ProgressFn = Callable[[str, int, int], None]


@dataclass(frozen=True)
class Param:
    """Describes one tunable setting; the GUI builds its input widget from this."""
    default: Union[int, float, str]
    help: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None


@dataclass
class DupGroup:
    """Images judged to be the same picture. Paths are relative to the data dir (e.g. raw/ffhq/ffhq_000001.png)."""
    members: List[str]
    score: float = 1.0          # weakest similarity linking the group (1.0 = identical)
    keep: Optional[List[str]] = None   # finder-chosen survivors (several allowed); None = pipeline keeps 1 (pick_keeper)


class DuplicateFinder(ABC):
    """Subclass this, set `name`, and decorate with @registry.register.

    A finder ONLY finds groups. Choosing which image to keep, moving files and updating the
    CSVs is done by the pipeline, so every back-end behaves the same way for the user.
    """
    name: str = ""
    description: str = ""
    requires: tuple = ()                         # importable module names, e.g. ("faiss", "torch")
    install_hint: str = ""                       # shown when a requirement is missing
    slow: bool = False                           # True = GUI asks for confirmation before a full (no --limit) run
    hidden: bool = False                         # True = not offered in the GUI (still usable from the CLI)
    params: Dict[str, Param] = {}

    def __init__(self, **overrides):
        unknown = set(overrides) - set(self.params)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {sorted(unknown)}; known: {sorted(self.params)}")
        self.p = {k: spec.default for k, spec in self.params.items()}
        for k, v in overrides.items():
            self.p[k] = type(self.params[k].default)(v)

    @classmethod
    def missing_requirements(cls) -> List[str]:
        return [m for m in cls.requires if find_spec(m) is None]

    @abstractmethod
    def find(self, data_dir: Path, images: List[str], progress: ProgressFn) -> List[DupGroup]:
        """Return groups with 2+ members. `images` are paths relative to `data_dir`."""
