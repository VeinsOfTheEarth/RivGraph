from __future__ import annotations

import pytest

from .utils import load_case


@pytest.fixture()
def delta_case_mossy():
    return load_case("delta_mossy")


@pytest.fixture()
def river_case_brahma_clipped():
    return load_case("river_brahma_clipped")
