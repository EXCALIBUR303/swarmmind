"""MAPPO / CTDE tests (§16, §12.4, §33.3).

The centrepiece is the parity test (§16.3): the ENTIRE MAPPO data path in
the trainer (agent-specific state in the vector env, global state in the
buffer, 9-channel assembly at rollout / truncation-bootstrap / update
time) switched on, with a LOCAL critic, must reproduce plain IPPO
bit-for-bit. That proves the global-state plumbing has no side effect on
RNG streams, ordering or the actor — so any IPPO/MAPPO difference in
results is attributable to the critic's input alone. A negative control
confirms the check isn't vacuous: with the real centralized critic the
parameters DO diverge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from marl.algorithms.common.config import AlgoConfig
from marl.algorithms.ippo.agent import IPPOAgent
from marl.algorithms.mappo.agent import MAPPOAgent
from marl.envs.core.config import EnvConfig, LevelRegistry
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.world import SwarmWorld
from marl.envs.vector import SyncVectorSwarm
from marl.training.checkpoint import load_checkpoint
from marl.training.trainer import _critic_inputs, train


def _algo(name: str) -> AlgoConfig:
    critic_input = "agent_specific_global" if name == "mappo" else "local"
    cfg = AlgoConfig.model_validate({"name": name, "critic": {"input": critic_input}})
    cfg.ppo.n_envs = 4
    cfg.ppo.rollout_len = 16
    cfg.eval.interval_updates = 1000
    cfg.eval.n_val_maps = 3
    cfg.checkpoint.fractions = [0.0, 1.0]
    return cfg


def _short_episode_l1() -> EnvConfig:
    # max_steps=8 forces several truncations within a 16-step rollout, so
    # the truncation-bootstrap path (which reads FINAL global state) is
    # exercised by every training test below.
    return LevelRegistry().get("L1").model_copy(update={"max_steps": 8})


class _IPPOWithGlobalDataPath(IPPOAgent):
    """IPPO's local critic, but with the trainer's MAPPO data path on."""

    uses_global_state = True


def _run(tmp_path: Path, name: str, builder, algo_cfg: AlgoConfig, steps: int, resume: bool = False):
    run_dir = tmp_path / name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return train(
        _short_episode_l1(), algo_cfg, builder, run_seed=5, total_env_steps=steps,
        run_dir=run_dir, study="t", algo_name=name, level="L1",
        resume_from=(run_dir / "checkpoints" / "latest.pt") if resume else None,
    ), run_dir


def _params(run_dir: Path, key: str) -> dict[str, torch.Tensor]:
    return load_checkpoint(run_dir / "checkpoints" / "latest.pt")[key]


# ---------------------------------------------------------------- construction


def test_mappo_agent_rejects_local_config():
    with pytest.raises(ValueError):
        MAPPOAgent(_algo("ippo"))


def test_mappo_critic_requires_global_state():
    from marl.algorithms.common.agent import CriticInputs

    agent = MAPPOAgent(_algo("mappo"))
    with pytest.raises(ValueError):
        agent.critic_value_denormalized(CriticInputs(torch.rand(2, 8, 11, 11), torch.rand(2, 10)))


def test_ippo_and_mappo_actors_identical_at_init():
    torch.manual_seed(123)
    ippo = IPPOAgent(_algo("ippo"))
    torch.manual_seed(123)
    mappo = MAPPOAgent(_algo("mappo"))
    for (ki, pi), (km, pm) in zip(
        ippo.actor.state_dict().items(), mappo.actor.state_dict().items(), strict=True
    ):
        assert ki == km
        assert torch.equal(pi, pm)


@pytest.mark.parametrize(
    "n_robots,size", [(2, 8), (3, 10), (4, 12)]
)
def test_mappo_critic_input_shapes_across_team_and_map_sizes(n_robots: int, size: int):
    env_cfg = EnvConfig(
        name="t", width=size, height=size, n_robots=n_robots, n_targets=1,
        obstacle_density=0.1, max_steps=50, sensor_radius=2.0, obs_radius=5,
    )
    e = 3
    vec = SyncVectorSwarm(env_cfg, n_envs=e, root_seed=0, include_agent_state=True)
    batch = vec.reset()
    assert batch.agent_state is not None
    assert batch.agent_state.shape == (e, n_robots, 2, size, size)

    grid = torch.as_tensor(batch.obs_grid, dtype=torch.float32).reshape(e * n_robots, 8, 11, 11)
    v = torch.as_tensor(batch.obs_vec, dtype=torch.float32).reshape(e * n_robots, 10)
    inputs = _critic_inputs(grid, v, batch.state_grid, batch.agent_state, batch.state_vec)
    assert inputs.state_grid is not None and inputs.state_vec is not None
    assert inputs.state_grid.shape == (e * n_robots, 9, size, size)
    assert inputs.state_vec.shape == (e * n_robots, 4)

    agent = MAPPOAgent(_algo("mappo"))
    mask = torch.as_tensor(batch.mask, dtype=torch.bool).reshape(e * n_robots, 5)
    actions, logp, value = agent.collect(grid, v, mask, inputs, torch.Generator().manual_seed(0))
    assert actions.shape == logp.shape == value.shape == (e * n_robots,)


def test_agent_specific_channels_identify_each_robot():
    """Channel 7 (own position) must differ per robot and match its true
    position — without it the critic can't tell agents apart (§16.8 item 11)."""
    env_cfg = LevelRegistry().get("L2")
    vec = SyncVectorSwarm(env_cfg, n_envs=2, root_seed=1, include_agent_state=True)
    batch = vec.reset()
    assert batch.agent_state is not None
    for e, world in enumerate(vec.worlds):
        assert world.state is not None
        for i in range(env_cfg.n_robots):
            y, x = world.state.robot_pos[i]
            pos_ch = batch.agent_state[e, i, 0]
            assert pos_ch[y, x] == 1.0 and pos_ch.sum() == 1.0


# ------------------------------------------------------ CTDE information boundary


def test_mappo_critic_sees_hidden_target_but_actor_does_not():
    """Reverse-direction sanity (§12.4): moving a HIDDEN target changes the
    centralized critic's value but leaves the actor's policy bit-identical."""
    cfg = EnvConfig(
        name="t", width=12, height=12, n_robots=1, n_targets=1, obstacle_density=0.0,
        max_steps=50, sensor_radius=1.0, obs_radius=5, comm_mode="none", target_min_dist=1,
    )
    rows = ["." * 12 for _ in range(12)]
    worlds = []
    for target in [(10, 10), (2, 10)]:
        w = SwarmWorld(cfg)
        w.reset(map_spec=MapSpec(id="m", width=12, height=12, terrain_rows=rows,
                                 robot_starts=[(1, 1)], targets=[target]))
        assert w.state is not None and w.state.target_status[0] == 0  # HIDDEN in both
        worlds.append(w)

    torch.manual_seed(0)
    agent = MAPPOAgent(_algo("mappo"))
    values, probs = [], []
    for w in worlds:
        g, v, m = w.all_actor_obs()
        sg, sv = w.global_state()
        agent_state = np.stack([w.agent_state_channels(0)])
        gt = torch.as_tensor(g, dtype=torch.float32)
        vt = torch.as_tensor(v, dtype=torch.float32)
        inputs = _critic_inputs(gt, vt, sg, agent_state, sv)
        with torch.no_grad():
            values.append(float(agent.critic_value_denormalized(inputs)[0]))
            probs.append(agent.actor(gt, vt, torch.as_tensor(m, dtype=torch.bool)).probs)
    assert torch.equal(probs[0], probs[1]), "actor output changed with a hidden target — LEAK"
    assert values[0] != values[1], "critic can't see the hidden target — global state isn't reaching it"


# ------------------------------------------------------------------- parity


def test_parity_mappo_data_path_with_local_critic_equals_ippo_bitwise(tmp_path):
    steps = 4 * 16 * 2  # 2 updates, with truncations (max_steps=8)
    _, ippo_dir = _run(tmp_path, "ippo", IPPOAgent, _algo("ippo"), steps)
    _, path_dir = _run(tmp_path, "ippo_globalpath", _IPPOWithGlobalDataPath, _algo("ippo"), steps)
    for key in ("actor_state_dict", "critic_state_dict"):
        a, b = _params(ippo_dir, key), _params(path_dir, key)
        for name in a:
            assert torch.equal(a[name], b[name]), f"{key}.{name} diverged — MAPPO data path has a side effect"


def test_parity_negative_control_real_central_critic_diverges(tmp_path):
    """Non-vacuity check: with the actual centralized critic, the actor's
    parameters after 2 updates MUST differ from IPPO's (the critic reaches
    the actor through the advantages). If this ever passes as equal, the
    parity test above would be proving nothing."""
    steps = 4 * 16 * 2
    _, ippo_dir = _run(tmp_path, "ippo", IPPOAgent, _algo("ippo"), steps)
    _, mappo_dir = _run(tmp_path, "mappo", MAPPOAgent, _algo("mappo"), steps)
    a, b = _params(ippo_dir, "actor_state_dict"), _params(mappo_dir, "actor_state_dict")
    assert any(not torch.equal(a[n], b[n]) for n in a)


def test_mappo_resume_equivalence_bitwise(tmp_path):
    """Resume must also restore everything MAPPO's critic reads (the live
    env states that the agent-specific channels are rebuilt from)."""
    cfg = _algo("mappo")
    full = 4 * 16 * 4
    _, straight = _run(tmp_path, "straight", MAPPOAgent, cfg, full)
    _run(tmp_path, "resumed", MAPPOAgent, cfg, full // 2)
    _, resumed = _run(tmp_path, "resumed", MAPPOAgent, cfg, full, resume=True)
    for key in ("actor_state_dict", "critic_state_dict"):
        a, b = _params(straight, key), _params(resumed, key)
        for name in a:
            assert torch.equal(a[name], b[name]), f"{key}.{name}"
