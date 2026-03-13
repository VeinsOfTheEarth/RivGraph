from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests._helpers import REGRESSION_DATA_ROOT, load_case_config


@dataclass(frozen=True)
class RegressionCase:
    key: str
    case_dir: Path
    config: dict[str, Any]

    @property
    def kind(self) -> str:
        return self.config["kind"]

    @property
    def name(self) -> str:
        return self.config["name"]

    def input_path(self, key: str) -> Path | None:
        rel = self.config.get("inputs", {}).get(key)
        if rel is None:
            return None
        return self.case_dir / rel

    def param(self, key: str, default: Any = None) -> Any:
        return self.config.get("params", {}).get(key, default)

    def check(self, key: str, default: Any = None) -> Any:
        return self.config.get("checks", {}).get(key, default)

    def expectation(self, key: str, default: Any = None) -> Any:
        return self.config.get("expectations", {}).get(key, default)


def load_case(case_key: str) -> RegressionCase:
    case_dir = REGRESSION_DATA_ROOT / case_key
    config = load_case_config(case_dir)
    return RegressionCase(key=case_key, case_dir=case_dir, config=config)
