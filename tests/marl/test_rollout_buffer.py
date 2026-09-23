"""RolloutBuffer layout/order and flatten tests (§15.4, §16.5, §33.3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from marl.buffers.rollout_buffer import RolloutBuffer, flatten_batch


def _make_filled_buffer(t=3, e=2, n=2, k=11) -> RolloutBuffer:
    buf = RolloutBuffer(t_len=t, n_envs=e, n_agents=n, grid_size=k)
    for step in range(t):
        truncated = np.zeros(e, dtype=bool)
        if e > 1:
            truncated[1] = step == t - 1  # env 1 truncates at the last step
        buf.add(
            obs_grid=np.full((e, n, 8, k, k), step, dtype=np.uint8),
            obs_vec=np.full((e, n, 10), step, dtype=np.float32),
            mask=np.ones((e, n, 5), dtype=np.int8),
            action=np.full((e, n), step, dtype=np.int64),
            logp_old=np.full((e, n), -float(step), dtype=np.float32),
            value_old=np.full((e, n), float(step), dtype=np.float32),
            reward=np.full((e, n), float(step), dtype=np.float32),
            terminated=np.zeros(e, dtype=bool),
            truncated=truncated,
        )
    return buf


def test_buffer_layout_shapes():
    buf = _make_filled_buffer(t=4, e=3, n=2, k=11)
    assert buf.obs_grid.shape == (4, 3, 2, 8, 11, 11)
    assert buf.obs_vec.shape == (4, 3, 2, 10)
    assert buf.mask.shape == (4, 3, 2, 5)
    assert buf.action.shape == (4, 3, 2)


def test_buffer_add_preserves_time_order():
    buf = _make_filled_buffer(t=3, e=1, n=1)
    assert buf.action[0, 0, 0] == 0
    assert buf.action[1, 0, 0] == 1
    assert buf.action[2, 0, 0] == 2


def test_buffer_raises_when_full():
    buf = _make_filled_buffer(t=2, e=1, n=1)
    with pytest.raises(IndexError):
        buf.add(
            obs_grid=np.zeros((1, 1, 8, 11, 11), dtype=np.uint8),
            obs_vec=np.zeros((1, 1, 10), dtype=np.float32),
            mask=np.ones((1, 1, 5), dtype=np.int8),
            action=np.zeros((1, 1), dtype=np.int64),
            logp_old=np.zeros((1, 1), dtype=np.float32),
            value_old=np.zeros((1, 1), dtype=np.float32),
            reward=np.zeros((1, 1), dtype=np.float32),
            terminated=np.zeros(1, dtype=bool),
            truncated=np.zeros(1, dtype=bool),
        )


def test_buffer_done_is_terminated_or_truncated():
    buf = _make_filled_buffer(t=3, e=2, n=2)
    done = buf.done
    assert done.shape == (3, 2)
    assert done[2, 1]  # env 1 truncated at the last step
    assert not done[0, 1]
    assert not done[:, 0].any()  # env 0 never terminates/truncates in this fixture


def test_to_torch_preserves_shapes_and_values():
    buf = _make_filled_buffer(t=2, e=2, n=2)
    t = buf.to_torch()
    assert t.obs_grid.shape == (2, 2, 2, 8, 11, 11)
    assert torch.equal(t.action, torch.as_tensor(buf.action))
    assert t.obs_grid.dtype == torch.float32  # uint8 -> float32 conversion


def test_flatten_batch_shape_is_t_times_e_times_n():
    buf = _make_filled_buffer(t=3, e=2, n=2)
    bt = buf.to_torch()
    adv = torch.zeros_like(bt.reward)
    ret = torch.zeros_like(bt.reward)
    flat = flatten_batch(bt, adv, ret)
    assert len(flat) == 3 * 2 * 2
    assert flat.obs_grid.shape == (12, 8, 11, 11)
    assert flat.action.shape == (12,)


def test_flatten_batch_preserves_sample_identity():
    """Flattening must not scramble which (t,e,n) each sample came from —
    check by encoding (t,e,n) into the action value and unflattening logic."""
    t, e, n = 2, 2, 2
    buf = RolloutBuffer(t_len=t, n_envs=e, n_agents=n, grid_size=11)
    for step in range(t):
        actions = np.zeros((e, n), dtype=np.int64)
        for ei in range(e):
            for ni in range(n):
                actions[ei, ni] = step * 100 + ei * 10 + ni
        buf.add(
            obs_grid=np.zeros((e, n, 8, 11, 11), dtype=np.uint8),
            obs_vec=np.zeros((e, n, 10), dtype=np.float32),
            mask=np.ones((e, n, 5), dtype=np.int8),
            action=actions,
            logp_old=np.zeros((e, n), dtype=np.float32),
            value_old=np.zeros((e, n), dtype=np.float32),
            reward=np.zeros((e, n), dtype=np.float32),
            terminated=np.zeros(e, dtype=bool),
            truncated=np.zeros(e, dtype=bool),
        )
    bt = buf.to_torch()
    adv = torch.zeros_like(bt.reward)
    ret = torch.zeros_like(bt.reward)
    flat = flatten_batch(bt, adv, ret)
    # Reconstruct expected flattened order: reshape(-1) on a [T,E,N] array
    # iterates T slowest, then E, then N fastest.
    expected = bt.action.reshape(-1)
    assert torch.equal(flat.action, expected)


def test_global_state_storage_optional():
    """When global_channels is None (IPPO), state_grid/state_vec stay None."""
    buf = RolloutBuffer(t_len=2, n_envs=1, n_agents=1, grid_size=11)
    assert buf.state_grid is None
    assert buf.state_vec is None


def test_global_state_storage_when_enabled_requires_state_args():
    buf = RolloutBuffer(t_len=1, n_envs=1, n_agents=1, grid_size=11, global_channels=9)
    with pytest.raises(ValueError):
        buf.add(
            obs_grid=np.zeros((1, 1, 8, 11, 11), dtype=np.uint8),
            obs_vec=np.zeros((1, 1, 10), dtype=np.float32),
            mask=np.ones((1, 1, 5), dtype=np.int8),
            action=np.zeros((1, 1), dtype=np.int64),
            logp_old=np.zeros((1, 1), dtype=np.float32),
            value_old=np.zeros((1, 1), dtype=np.float32),
            reward=np.zeros((1, 1), dtype=np.float32),
            terminated=np.zeros(1, dtype=bool),
            truncated=np.zeros(1, dtype=bool),
            # state_grid/state_vec omitted -> should raise
        )
