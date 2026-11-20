"""Finder registry. Finders in tools/dedup/finders/*.py register themselves; nothing else to edit."""
import importlib
import pkgutil
from typing import Dict, Type

from .base import DuplicateFinder

_FINDERS: Dict[str, Type[DuplicateFinder]] = {}
_LOAD_ERRORS: Dict[str, str] = {}
_loaded = False


def register(cls: Type[DuplicateFinder]) -> Type[DuplicateFinder]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a name")
    _FINDERS[cls.name] = cls
    return cls


def _load_all() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import finders
    for mod in pkgutil.iter_modules(finders.__path__):
        if mod.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{finders.__name__}.{mod.name}")
        except Exception as e:  # a broken plugin must not break the others
            _LOAD_ERRORS[mod.name] = f"{type(e).__name__}: {e}"


def available() -> Dict[str, Type[DuplicateFinder]]:
    _load_all()
    return dict(_FINDERS)


def load_errors() -> Dict[str, str]:
    _load_all()
    return dict(_LOAD_ERRORS)


def create(name: str, **params) -> DuplicateFinder:
    finders = available()
    if name not in finders:
        raise KeyError(f"unknown finder '{name}'. available: {sorted(finders)}")
    cls = finders[name]
    missing = cls.missing_requirements()
    if missing:
        raise RuntimeError(f"finder '{name}' needs {missing}. {cls.install_hint}")
    return cls(**params)
