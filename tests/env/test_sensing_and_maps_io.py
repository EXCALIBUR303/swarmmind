"""Sensing geometry and MapSpec serialization tests (§11.2, §20.2)."""

from __future__ import annotations

import pytest

from marl.envs.core.maps_io import MapSpec, load_map, load_structured_map_txt, save_map
from marl.envs.core.sensing import sensed_cells, sensed_mask


def test_sensed_cells_radius_one_is_4_neighbourhood_plus_centre():
    cells = sensed_cells((5, 5), 1.0, 20, 20)
    cell_set = {tuple(c) for c in cells}
    assert cell_set == {(5, 5), (4, 5), (6, 5), (5, 4), (5, 6)}


def test_sensed_cells_clips_to_bounds():
    cells = sensed_cells((0, 0), 1.0, 10, 10)
    cell_set = {tuple(c) for c in cells}
    assert (-1, 0) not in cell_set
    assert (0, -1) not in cell_set
    assert (0, 0) in cell_set


def test_sensed_mask_matches_sensed_cells():
    mask = sensed_mask((3, 3), 2.0, 10, 10)
    cells = sensed_cells((3, 3), 2.0, 10, 10)
    assert mask.sum() == len(cells)
    for y, x in cells:
        assert mask[y, x]


def test_sensed_cells_larger_radius_is_superset():
    small = {tuple(c) for c in sensed_cells((5, 5), 1.0, 20, 20)}
    large = {tuple(c) for c in sensed_cells((5, 5), 2.0, 20, 20)}
    assert small.issubset(large)
    assert len(large) > len(small)


def test_mapspec_roundtrip_save_load(tmp_path):
    m = MapSpec(
        id="test1",
        width=4,
        height=3,
        terrain_rows=["....", ".##.", "...."],
        robot_starts=[(0, 0), (2, 3)],
        targets=[(0, 3)],
        generator={"kind": "handmade"},
    )
    path = tmp_path / "map.json"
    save_map(m, path)
    loaded = load_map(path)
    assert loaded.terrain_rows == m.terrain_rows
    assert loaded.robot_starts == m.robot_starts
    assert loaded.targets == m.targets
    assert loaded.content_hash == m.content_hash


def test_mapspec_rejects_mismatched_row_count():
    with pytest.raises(ValueError):
        MapSpec(id="bad", width=4, height=3, terrain_rows=["....", "...."], robot_starts=[], targets=[])


def test_mapspec_rejects_mismatched_row_width():
    with pytest.raises(ValueError):
        MapSpec(id="bad", width=4, height=2, terrain_rows=["....", "..."], robot_starts=[], targets=[])


def test_mapspec_content_hash_ignores_id():
    m1 = MapSpec(id="a", width=3, height=3, terrain_rows=["...", "...", "..."], robot_starts=[(0, 0)], targets=[(2, 2)])
    m2 = MapSpec(id="b", width=3, height=3, terrain_rows=["...", "...", "..."], robot_starts=[(0, 0)], targets=[(2, 2)])
    assert m1.content_hash == m2.content_hash


def test_mapspec_content_hash_differs_on_terrain_change():
    m1 = MapSpec(id="a", width=3, height=3, terrain_rows=["...", "...", "..."], robot_starts=[(0, 0)], targets=[(2, 2)])
    m2 = MapSpec(id="a", width=3, height=3, terrain_rows=["...", ".#.", "..."], robot_starts=[(0, 0)], targets=[(2, 2)])
    assert m1.content_hash != m2.content_hash


def test_terrain_array_roundtrip():
    m = MapSpec(id="a", width=3, height=2, terrain_rows=[".#.", "..."], robot_starts=[(0, 0)], targets=[(1, 2)])
    arr = m.terrain_array()
    assert arr.shape == (2, 3)
    assert arr[0, 1] == 1  # obstacle
    assert arr[0, 0] == 0

    m2 = MapSpec.from_terrain_array("b", arr, [(0, 0)], [(1, 2)])
    assert m2.terrain_rows == m.terrain_rows


def test_load_structured_map_txt(tmp_path):
    txt = "R.T\n.#.\n...\n"
    path = tmp_path / "structured.txt"
    path.write_text(txt)
    m = load_structured_map_txt(path, map_id="mymap")
    assert m.id == "mymap"
    assert m.width == 3
    assert m.height == 3
    assert m.robot_starts == [(0, 0)]
    assert m.targets == [(0, 2)]
    assert m.terrain_rows[1][1] == "#"


def test_load_structured_map_rejects_inconsistent_widths(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("...\n..\n")
    with pytest.raises(ValueError):
        load_structured_map_txt(path)
