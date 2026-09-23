"""Actor/critic network shape and masking tests (§15.2, §33.3)."""

from __future__ import annotations

import torch

from marl.networks.actor import ActorPolicy
from marl.networks.critics import CentralCritic, LocalCritic
from marl.networks.encoders import LocalEncoder


def test_local_encoder_output_shape():
    enc = LocalEncoder(grid_size=11)
    grid = torch.rand(3, 8, 11, 11)
    vec = torch.rand(3, 10)
    h = enc(grid, vec)
    assert h.shape == (3, 256)


def test_local_encoder_conv_flatten_matches_spec_2304():
    """The spec derives its exact hidden-layer sizes from a 2304 = 64*6*6
    flatten at K=11 (§15.2). Confirms the architecture matches."""
    enc = LocalEncoder(grid_size=11)
    dummy = torch.zeros(1, 8, 11, 11)
    conv_out = enc.conv(dummy)
    assert conv_out.numel() == 64 * 6 * 6


def test_actor_forward_signature_only_grid_vec_mask():
    actor = ActorPolicy()
    grid = torch.rand(2, 8, 11, 11)
    vec = torch.rand(2, 10)
    mask = torch.ones(2, 5, dtype=torch.bool)
    dist = actor(grid, vec, mask)
    assert dist.probs.shape == (2, 5)


def test_actor_never_samples_masked_action():
    actor = ActorPolicy()
    grid = torch.rand(1, 8, 11, 11)
    vec = torch.rand(1, 10)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    mask[0, 3] = True
    gen = torch.Generator().manual_seed(0)
    for _ in range(30):
        actions, _ = actor.act(grid, vec, mask, generator=gen, stochastic=True)
        assert actions.item() == 3


def test_actor_greedy_mode_is_deterministic():
    actor = ActorPolicy()
    actor.eval()
    grid = torch.rand(1, 8, 11, 11)
    vec = torch.rand(1, 10)
    mask = torch.ones(1, 5, dtype=torch.bool)
    a1, _ = actor.act(grid, vec, mask, stochastic=False)
    a2, _ = actor.act(grid, vec, mask, stochastic=False)
    assert a1.item() == a2.item()


def test_actor_initial_policy_near_uniform_over_valid_actions():
    """Policy head init gain 0.01 -> near-uniform initial distribution (§15.2)."""
    actor = ActorPolicy()
    grid = torch.rand(1, 8, 11, 11)
    vec = torch.rand(1, 10)
    mask = torch.ones(1, 5, dtype=torch.bool)
    dist = actor(grid, vec, mask)
    probs = dist.probs[0]
    assert torch.allclose(probs, torch.full((5,), 0.2), atol=0.05)


def test_local_critic_output_shape():
    critic = LocalCritic()
    grid = torch.rand(4, 8, 11, 11)
    vec = torch.rand(4, 10)
    v = critic(grid, vec)
    assert v.shape == (4,)


def test_actor_and_critic_do_not_share_weights():
    """No weight SHARING (object identity) — separate parameter tensors,
    not separate but coincidentally-equal values (biases are legitimately
    zero-initialized in both, so a value-equality check would false-fail)."""
    actor = ActorPolicy()
    critic = LocalCritic()
    actor_params = list(actor.encoder.parameters())
    critic_params = list(critic.encoder.parameters())
    assert all(a is not c for a, c in zip(actor_params, critic_params, strict=True))
    # And the learned (non-zero-initialized) weight tensors DO differ in
    # value too, since orthogonal init is randomized independently.
    actor_weights = [p for p in actor_params if p.dim() > 1]
    critic_weights = [p for p in critic_params if p.dim() > 1]
    assert all(
        not torch.equal(a, c) for a, c in zip(actor_weights, critic_weights, strict=True)
    )


def test_central_critic_output_shape_and_accepts_any_map_size():
    critic = CentralCritic(global_channels=9, grid_size=11)
    for h, w in [(8, 8), (10, 10), (12, 14)]:
        global_grid = torch.rand(2, 9, h, w)
        global_vec = torch.rand(2, 4)
        local_grid = torch.rand(2, 8, 11, 11)
        local_vec = torch.rand(2, 10)
        v = critic(global_grid, global_vec, local_grid, local_vec)
        assert v.shape == (2,)


def test_networks_param_count_matches_spec_estimate():
    """Spec §15.2 estimates ~0.73M parameters per encoder-based network."""
    actor = ActorPolicy()
    n = sum(p.numel() for p in actor.parameters())
    assert 700_000 < n < 760_000
