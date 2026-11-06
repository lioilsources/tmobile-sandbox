"""Shared fixtures: a small synthetic store (2 leagues x 3 seasons x 8 teams), built once per session."""
import pytest

import synth


@pytest.fixture(scope="session")
def small_db(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("uc26") / "small.sqlite")
    synth.generate(path, edge=0.5, seed=7, n_leagues=2, n_seasons=3, n_teams=8)
    return path
