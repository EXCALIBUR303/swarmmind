"""Config loading, override, and validation tests (§32)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from marl.envs.core.config import EnvConfig, LevelRegistry
from marl.utils.config_loader import apply_overrides, load_config


def _minimal_env_dict(**overrides) -> dict:
    base = dict(
        name="test",
        width=10,
        height=10,
        n_robots=2,
        n_targets=1,
        obstacle_density=0.1,
        max_steps=100,
        sensor_radius=2.0,
        obs_radius=5,
    )
    base.update(overrides)
    return base


def test_env_config_valid_minimal():
    cfg = EnvConfig.model_validate(_minimal_env_dict())
    assert cfg.name == "test"
    assert cfg.comm_mode == "shared_map"
    assert cfg.reward.rescue == 1.0


def test_env_config_capacity_check_accepts_tight_but_valid_config():
    # At the smallest allowed grid (6x6=36 cells) with max density (0.35),
    # the occupiable capacity is floor(36*0.65)=23, comfortably above the
    # largest allowed n_robots+n_targets (6+5=11). The capacity guard can
    # therefore never trigger within the current field bounds — it exists
    # as a defensive check for if those bounds are loosened later. This
    # test documents that boundary case still validates cleanly.
    cfg = EnvConfig.model_validate(
        _minimal_env_dict(width=6, height=6, n_robots=6, n_targets=5, obstacle_density=0.35)
    )
    assert cfg.n_robots == 6
    assert cfg.n_targets == 5


def test_env_config_rejects_obs_radius_smaller_than_sensor_radius():
    with pytest.raises(ValidationError):
        EnvConfig.model_validate(_minimal_env_dict(obs_radius=1, sensor_radius=2.0))


def test_env_config_rejects_bad_density():
    with pytest.raises(ValidationError):
        EnvConfig.model_validate(_minimal_env_dict(obstacle_density=0.5))


def test_env_config_rejects_too_many_robots():
    with pytest.raises(ValidationError):
        EnvConfig.model_validate(_minimal_env_dict(n_robots=7))


def test_env_config_limited_range_requires_comm_radius():
    with pytest.raises(ValidationError):
        EnvConfig.model_validate(
            _minimal_env_dict(comm_mode="limited_range", comm_radius=0.0)
        )


def test_level_registry_loads_all_levels():
    registry = LevelRegistry()
    levels = registry.all()
    assert set(levels.keys()) == {"L1", "L2", "L3", "L4"}
    assert levels["L2"].n_robots == 3
    assert levels["L2"].n_targets == 2


def test_level_registry_unknown_level_raises():
    registry = LevelRegistry()
    with pytest.raises(KeyError):
        registry.get("L99")


def test_level_registry_caches():
    registry = LevelRegistry()
    a = registry.get("L1")
    b = registry.get("L1")
    assert a is b


def test_apply_overrides_type_coercion():
    base = {"a": {"b": 1, "c": 0.5, "d": True, "e": "hello"}}
    out = apply_overrides(base, ["a.b=5", "a.c=1.5", "a.d=false", "a.e=world"])
    assert out["a"]["b"] == 5
    assert isinstance(out["a"]["b"], int)
    assert out["a"]["c"] == 1.5
    assert out["a"]["d"] is False
    assert out["a"]["e"] == "world"


def test_apply_overrides_rejects_unknown_key():
    base = {"a": {"b": 1}}
    with pytest.raises(KeyError):
        apply_overrides(base, ["a.z=5"])


def test_apply_overrides_rejects_malformed():
    with pytest.raises(ValueError):
        apply_overrides({"a": 1}, ["no-equals-sign"])


def test_load_config_from_file(tmp_path):
    p = tmp_path / "env.yaml"
    p.write_text(
        "name: t\nwidth: 8\nheight: 8\nn_robots: 2\nn_targets: 1\n"
        "obstacle_density: 0.1\nmax_steps: 50\nsensor_radius: 2.0\nobs_radius: 5\n"
    )
    cfg = load_config(p, EnvConfig, overrides=["n_robots=3"])
    assert cfg.n_robots == 3


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml", EnvConfig)
