from __future__ import annotations

import warnings


def pytest_configure(config):
    # Upstream pyogrio<fixed-version imports shapely.geos at import time, which
    # triggers a deprecation warning under modern Shapely. This is third-party
    # noise rather than a RivGraph warning regression.
    warnings.filterwarnings(
        "ignore",
        message=r"The 'shapely\.geos' module is deprecated, and will be removed in a future version\..*",
        category=DeprecationWarning,
        module=r"pyogrio.*",
    )
