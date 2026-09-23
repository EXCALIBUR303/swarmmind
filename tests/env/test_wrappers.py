"""PettingZoo wrapper conformance and PettingZoo <-> vector wrapper parity (§7, §33.2)."""

from __future__ import annotations

import numpy as np
from pettingzoo.test import parallel_api_test

from marl.envs.core.config import EnvConfig
from marl.envs.pettingzoo_env import SwarmParallelEnv
from marl.envs.vector import SyncVectorSwarm


def _cfg(**overrides) -> EnvConfig:
    base = dict(
        name="t",
        width=8,
        height=8,
        n_robots=2,
        n_targets=2,
        obstacle_density=0.1,
        max_steps=40,
        sensor_radius=2.0,
        obs_radius=5,
    )
    base.update(overrides)
    return EnvConfig.model_validate(base)


def test_parallel_api_conformance():
    env = SwarmParallelEnv(_cfg())
    parallel_api_test(env, num_cycles=150)


def test_pettingzoo_reset_deterministic_given_seed():
    env1 = SwarmParallelEnv(_cfg())
    env2 = SwarmParallelEnv(_cfg())
    obs1, _ = env1.reset(seed=77)
    obs2, _ = env2.reset(seed=77)
    for name in env1.possible_agents:
        assert np.array_equal(obs1[name]["grid"], obs2[name]["grid"])
        assert np.array_equal(obs1[name]["vec"], obs2[name]["vec"])


def test_pettingzoo_episode_ends_agents_empty():
    cfg = _cfg(width=6, height=6, n_robots=1, n_targets=1, max_steps=2)
    env = SwarmParallelEnv(cfg)
    env.reset(seed=1)
    for _ in range(2):
        actions = {name: 0 for name in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
    assert env.agents == []


def test_pettingzoo_state_matches_world_global_state():
    env = SwarmParallelEnv(_cfg())
    env.reset(seed=3)
    state = env.state()
    shared_grid, _ = env.world.global_state()
    assert np.array_equal(state, shared_grid)


def test_pettingzoo_vector_parity_same_seed_same_actions():
    """The SAME (map, seed, actions) must produce identical observations,
    rewards and termination flags through BOTH the PettingZoo dict path
    and the vectorized batch path (§7.1 single-code-path guarantee)."""
    cfg = _cfg(width=8, height=8, n_robots=3, n_targets=2, max_steps=40)
    seed = 55

    pz_env = SwarmParallelEnv(cfg)
    pz_env.reset(seed=seed)

    vec_env = SyncVectorSwarm(cfg, n_envs=1, root_seed=0)
    # Force the vector env's single world to reset onto the SAME map/seed
    # as the PettingZoo env, bypassing its own seed-generator stream.
    vec_env.worlds[0].reset(seed=seed)
    vec_env._reset_done = True

    rng = np.random.default_rng(11)
    for _ in range(30):
        actions_arr = rng.integers(0, 5, size=cfg.n_robots)
        pz_actions = {name: int(a) for name, a in zip(pz_env.agents, actions_arr, strict=True)}
        if not pz_env.agents:
            break

        pz_obs, pz_rewards, pz_terms, pz_truncs, _ = pz_env.step(pz_actions)
        vec_batch = vec_env.step(actions_arr.reshape(1, -1))

        for i, name in enumerate(f"robot_{j}" for j in range(cfg.n_robots)):
            if name not in pz_obs:
                continue
            assert np.array_equal(pz_obs[name]["grid"], vec_batch.obs_grid[0, i])
            assert np.array_equal(pz_obs[name]["vec"], vec_batch.obs_vec[0, i])
            assert abs(pz_rewards[name] - float(vec_batch.reward[0, i])) < 1e-6

        if pz_terms.get("robot_0") or pz_truncs.get("robot_0"):
            assert vec_batch.terminated[0] == pz_terms["robot_0"]
            assert vec_batch.truncated[0] == pz_truncs["robot_0"]
            break


def test_vector_env_batch_shapes():
    cfg = _cfg(n_robots=2, n_targets=2)
    vec = SyncVectorSwarm(cfg, n_envs=3, root_seed=1)
    batch = vec.reset()
    k = 2 * cfg.obs_radius + 1
    assert batch.obs_grid.shape == (3, 2, 8, k, k)
    assert batch.obs_vec.shape == (3, 2, 10)
    assert batch.mask.shape == (3, 2, 5)
    assert batch.state_grid.shape == (3, 7, cfg.height, cfg.width)
    assert batch.state_vec.shape == (3, 4)
    assert batch.final is None


def test_vector_env_auto_reset_returns_new_episode_obs():
    cfg = _cfg(width=6, height=6, n_robots=1, n_targets=1, max_steps=2)
    vec = SyncVectorSwarm(cfg, n_envs=2, root_seed=1)
    vec.reset()
    map_ids_before = [w.state.map_id for w in vec.worlds]
    for _ in range(2):
        batch = vec.step(np.zeros((2, 1), dtype=np.int32))
    # After 2 steps at max_steps=2, both envs should have truncated and
    # auto-reset onto a NEW map.
    map_ids_after = [w.state.map_id for w in vec.worlds]
    assert map_ids_after != map_ids_before
    assert batch.final is not None
    assert set(batch.final.env_indices) == {0, 1}


def test_vector_env_wrong_action_shape_raises():
    cfg = _cfg(n_robots=2)
    vec = SyncVectorSwarm(cfg, n_envs=3, root_seed=1)
    vec.reset()
    import pytest

    with pytest.raises(ValueError):
        vec.step(np.zeros((3, 3), dtype=np.int32))  # wrong n_robots dim


def test_pettingzoo_state_space_and_render_and_close():
    cfg = _cfg()
    env = SwarmParallelEnv(cfg, render_mode="ansi")
    env.reset(seed=1)
    assert env.state_space.shape == env.state().shape
    rendered = env.render()
    assert isinstance(rendered, str)
    assert "\n" in rendered
    env.close()  # no-op, must not raise

    env_no_render = SwarmParallelEnv(cfg)  # render_mode=None
    env_no_render.reset(seed=1)
    assert env_no_render.render() is None


def test_pettingzoo_step_with_no_active_agents_raises():
    cfg = _cfg(width=6, height=6, n_robots=1, n_targets=1, max_steps=1)
    env = SwarmParallelEnv(cfg)
    env.reset(seed=1)
    env.step({name: 0 for name in env.agents})  # ends the episode (max_steps=1)
    assert env.agents == []
    import pytest

    with pytest.raises(RuntimeError):
        env.step({})


def test_vector_env_step_before_reset_raises():
    cfg = _cfg()
    vec = SyncVectorSwarm(cfg, n_envs=2, root_seed=1)
    import pytest

    with pytest.raises(RuntimeError):
        vec.step(np.zeros((2, cfg.n_robots), dtype=np.int32))
