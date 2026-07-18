from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import TypeVar


T = TypeVar("T")


def _load_module(module_reference: str) -> ModuleType:
    candidate = Path(module_reference).expanduser()
    if candidate.suffix == ".py" or candidate.is_file():
        path = candidate.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"plugin module not found: {path}")
        digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
        module_name = f"sids_dynamic_plugin_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load plugin module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_reference)


def load_plugin(spec: str, expected_type: type[T]) -> T:
    """Load ``module:attribute`` or ``path.py:attribute`` and instantiate it."""

    module_reference, separator, attribute_name = spec.rpartition(":")
    if not separator or not module_reference or not attribute_name:
        raise ValueError(
            "plugin spec must have the form 'module:attribute' or "
            "'path.py:attribute'"
        )

    module = _load_module(module_reference)
    try:
        plugin_object = getattr(module, attribute_name)
    except AttributeError as error:
        raise AttributeError(
            f"plugin attribute {attribute_name!r} not found in {module_reference!r}"
        ) from error

    if isinstance(plugin_object, expected_type):
        instance = plugin_object
    elif inspect.isclass(plugin_object) or callable(plugin_object):
        instance = plugin_object()
    else:
        instance = plugin_object

    if not isinstance(instance, expected_type):
        raise TypeError(
            f"plugin {spec!r} produced {type(instance).__name__}; "
            f"expected {expected_type.__name__}"
        )
    return instance
