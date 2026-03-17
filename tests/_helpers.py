from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

TESTS_ROOT = Path(__file__).resolve().parent
REGRESSION_DATA_ROOT = TESTS_ROOT / "regression" / "data"


def require_gdal_bindings() -> None:
    """Legacy no-op retained for older tests during the Rasterio migration."""
    return


def require_rivgraph_classes():
    """Import RivGraph classes."""
    from rivgraph.classes import delta, river, rivnetwork

    return delta, river, rivnetwork


def require_io_utils():
    """Import io_utils."""
    from rivgraph import io_utils

    return io_utils


def require_rasters():
    """Import the future raster backend module when available."""
    import pytest
    return pytest.importorskip(
        'rivgraph.rasters',
        reason='raster backend refactor not yet implemented',
    )


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null" or lowered == "none":
        return None
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Minimal YAML loader for the small regression config files.

    Falls back when PyYAML is not installed. Supports nested mappings and
    simple scalar lists using two-space indentation.
    """
    lines = path.read_text(encoding="utf-8").splitlines()

    def next_meaningful_line(start_idx: int):
        for j in range(start_idx + 1, len(lines)):
            stripped = lines[j].strip()
            if stripped and not stripped.startswith("#"):
                return lines[j]
        return None

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for i, raw_line in enumerate(lines):
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        while indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]

        if stripped.startswith("- "):
            if not isinstance(container, list):
                raise ValueError(f"Malformed YAML list item in {path}: {raw_line}")
            container.append(_coerce_scalar(stripped[2:]))
            continue

        key, sep, remainder = stripped.partition(":")
        if not sep:
            raise ValueError(f"Malformed YAML line in {path}: {raw_line}")

        key = key.strip()
        remainder = remainder.strip()

        if remainder == "":
            next_line = next_meaningful_line(i)
            if next_line is None:
                new_container: Any = {}
            else:
                next_stripped = next_line.strip()
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                if next_indent <= indent:
                    new_container = {}
                elif next_stripped.startswith("- "):
                    new_container = []
                else:
                    new_container = {}
            container[key] = new_container
            stack.append((indent, new_container))
        else:
            container[key] = _coerce_scalar(remainder)

    return root


def load_case_config(case_dir: Path) -> dict[str, Any]:
    config_path = case_dir / "config.yml"
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _load_simple_yaml(config_path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))
