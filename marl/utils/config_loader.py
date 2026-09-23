"""Config loading, dotted-key overrides, and validation glue (§32).

This is intentionally small (~100 lines, per the spec) rather than pulling in
a framework like Hydra. It supports:

  - Loading a YAML file into a plain dict.
  - Applying dotted-key CLI overrides (`--set algo.lr_actor=1e-4`) with type
    coercion inferred from the existing value at that key.
  - Validating the merged dict into a pydantic model, surfacing errors
    clearly.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return data or {}


def _coerce(raw: str, reference: Any) -> Any:
    """Coerce a CLI override string to match the type of `reference`.

    Falls back to `ast.literal_eval` for anything that isn't a plain string
    at the reference location (so bools/ints/floats/lists/null all work),
    and falls back to the raw string if literal_eval fails (e.g. for actual
    string values like "shared_map").
    """
    if isinstance(reference, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(raw)
    if isinstance(reference, float):
        return float(raw)
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `["a.b.c=1", ...]`-style dotted overrides onto a copy of `config`.

    Raises KeyError if an intermediate key path doesn't already exist in
    `config` — overrides may only change existing keys, never silently
    introduce new ones (a common source of silent typos).
    """
    result: dict[str, Any] = {k: v for k, v in config.items()}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be of the form key.path=value, got: {item}")
        key_path, raw_value = item.split("=", 1)
        keys = key_path.split(".")
        node = result
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node:
                raise KeyError(f"unknown override key path: {key_path}")
            node = node[k]
        last = keys[-1]
        if not isinstance(node, dict) or last not in node:
            raise KeyError(f"unknown override key path: {key_path}")
        node[last] = _coerce(raw_value, node[last])
    return result


def validate(config: dict[str, Any], model: type[T]) -> T:
    """Validate `config` into `model`, raising a clear error on failure."""
    try:
        return model.model_validate(config)
    except ValidationError as e:
        raise ValueError(f"config validation failed for {model.__name__}:\n{e}") from e


def load_config(path: str | Path, model: type[T], overrides: list[str] | None = None) -> T:
    """Load a YAML file, apply overrides, and validate into `model`."""
    raw = load_yaml(path)
    if overrides:
        raw = apply_overrides(raw, overrides)
    return validate(raw, model)
