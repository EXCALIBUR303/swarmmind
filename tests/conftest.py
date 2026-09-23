"""Shared pytest fixtures: tiny configs and fixed seeds (§33.1)."""

from __future__ import annotations

import pytest


@pytest.fixture
def fixed_seed() -> int:
    return 12345
