"""The algorithm-agnostic PPO training loop (§15.9).

Works identically for IPPO and MAPPO — the only algorithm-specific piece
is which `PPOAgent` subclass is passed in (§16.3). Nothing here decides
between "local" and "agent_specific_global" critic input; that's entirely
inside the agent's `_critic_forward`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from marl.algorithms.common.agent import CriticInputs, PPOAgent
from marl.algorithms.common.config import AlgoConfig
from marl.algorithms.common.gae import compute_gae, normalize_advantages
from marl.algorithms.common.ppo_update import (
    clip_grad_norm,
    compute_update_losses,
    explained_variance,
)
from marl.buffers.rollout_buffer import RolloutBuffer, build_agent_specific_state, flatten_batch
from marl.envs.core.config import EnvConfig, LevelRegistry
from marl.envs.vector import SyncVectorSwarm
from marl.evaluation.eval_sets import frozen_hash_manifest
from marl.training.checkpoint import (
    export_actor,
    load_checkpoint,
    restore_agent,
    restore_rng_state,
    save_checkpoint,
)
from marl.training.logger import JsonlLogger
from marl.training.resume import restore_trainer_state, snapshot_trainer_state
from marl.training.run_dir import RunManifest
from marl.training.validate import evaluate_on_validation
from marl.utils.hashing import config_hash
from marl.utils.seeding import SeedStreams, seed_everything


@dataclass
class TrainResult:
    run_dir: Path
    global_env_steps: int
    global_agent_steps: int
    updates: int
    final_val_metrics: dict[str, Any] | None
    status: str
    checkpoints: list[str] = field(default_factory=list)


def _flatten_en(x: torch.Tensor, e: int, n: int) -> torch.Tensor:
    return x.reshape(e * n, *x.shape[2:])


def _unflatten_en(x: torch.Tensor, e: int, n: int) -> torch.Tensor:
    return x.reshape(e, n, *x.shape[1:])


def _critic_inputs(
    local_grid: torch.Tensor,  # [M, 8, K, K], M = (E*)N in env-major, agent-minor order
    local_vec: torch.Tensor,  # [M, 10]
    shared_grid: np.ndarray,  # [(E,) 7, H, W]
    agent_state: np.ndarray | None,  # [(E,) N, 2, H, W]; None -> local-only critic input (IPPO)
    state_vec: np.ndarray,  # [(E,) 4]
) -> CriticInputs:
    """Build the critic's inputs for M agent-samples. The single place the
    agent-specific global state is assembled at rollout / bootstrap time —
    it goes through the same `build_agent_specific_state` the update-time
    `flatten_batch` uses, so the channel layout can't diverge between them."""
    if agent_state is None:
        return CriticInputs(local_grid=local_grid, local_vec=local_vec)
    grid, vec = build_agent_specific_state(
        torch.as_tensor(shared_grid, dtype=torch.float32),
        torch.as_tensor(agent_state, dtype=torch.float32),
        torch.as_tensor(state_vec, dtype=torch.float32),
    )
    return CriticInputs(
        local_grid=local_grid,
        local_vec=local_vec,
        state_grid=grid.reshape(-1, *grid.shape[-3:]),
        state_vec=vec.reshape(-1, vec.shape[-1]),
    )


def train(
    env_cfg: EnvConfig,
    algo_cfg: AlgoConfig,
    agent_builder,  # Callable[[AlgoConfig, int], PPOAgent] — e.g. IPPOAgent
    run_seed: int,
    total_env_steps: int,
    run_dir: Path,
    study: str,
    algo_name: str,
    level: str,
    resume_from: Path | None = None,
) -> TrainResult:
    seed_everything(run_seed)
    streams = SeedStreams(run_seed)
    env_seed_gens = streams.spawn_generators(algo_cfg.ppo.n_envs)
    policy_gen = streams.spawn_torch_generator()
    shuffle_rng = streams.spawn_generators(1)[0]
    init_seed = streams.spawn_init_seed()

    torch.manual_seed(init_seed)
    grid_size = 2 * env_cfg.obs_radius + 1
    agent: PPOAgent = agent_builder(algo_cfg, grid_size)

    e_n, t_len, n_robots = algo_cfg.ppo.n_envs, algo_cfg.ppo.rollout_len, env_cfg.n_robots
    env_steps_per_update = e_n * t_len
    n_updates = max(total_env_steps // env_steps_per_update, 1)
    # Checkpoint fractions are fractions of the steps actually trained. The
    # budget is rarely a multiple of E*T (e.g. 3M / 2048), and measuring
    # against the nominal budget meant fraction 1.0 was never reached, so the
    # final checkpoint was silently missing from the schedule.
    trained_env_steps = n_updates * env_steps_per_update

    resolved_config = {"env": env_cfg.model_dump(), "algo": algo_cfg.model_dump(), "run_seed": run_seed}
    h8 = config_hash(resolved_config)
    if resume_from is not None:
        manifest = RunManifest.load(run_dir)
        manifest.data["status"] = "running"
        manifest.data["resumed_from"] = manifest.data.get("resumed_from", []) + [str(resume_from)]
        manifest.save()
    else:
        manifest = RunManifest.create(
            run_dir, study, algo_name, level, run_seed, env_cfg.model_dump(), algo_cfg.model_dump(), h8
        )
        manifest.data["budget_env_steps"] = total_env_steps
        manifest.data["trained_env_steps"] = trained_env_steps
        manifest.save()

    uses_global = agent.uses_global_state
    state_hw = (env_cfg.height, env_cfg.width) if uses_global else None
    # §20.3 isolation guard: no frozen validation/test map of ANY level may
    # appear in the training stream.
    eval_hashes = frozenset(frozen_hash_manifest(LevelRegistry().available_levels()))
    vec_env = SyncVectorSwarm(
        env_cfg,
        n_envs=e_n,
        seed_generators=env_seed_gens,
        include_agent_state=uses_global,
        exclude_hashes=eval_hashes,
    )
    batch = vec_env.reset()

    metrics_logger = JsonlLogger(run_dir / "metrics.jsonl")
    val_logger = JsonlLogger(run_dir / "eval_val.jsonl")

    global_env_steps = 0
    global_agent_steps = 0
    start_update = 0
    checkpoints_saved: set[float] = set()

    if resume_from is not None:
        ckpt = load_checkpoint(resume_from)
        restore_agent(agent, ckpt)
        restore_rng_state(ckpt)
        restore_trainer_state(ckpt["rng_state"], env_seed_gens, policy_gen, shuffle_rng, vec_env)
        batch = vec_env.current_batch()
        global_env_steps = ckpt["global_env_steps"]
        global_agent_steps = ckpt["global_agent_steps"] if ckpt.get("global_agent_steps") is not None else global_env_steps * n_robots
        start_update = ckpt["update_idx"] + 1
        frac_now = global_env_steps / trained_env_steps
        checkpoints_saved = {f for f in algo_cfg.checkpoint.fractions if frac_now + 1e-9 >= f}
    checkpoint_paths: list[str] = []
    final_val_metrics: dict[str, Any] | None = None

    def _trainer_rng_snapshot() -> dict[str, Any]:
        return snapshot_trainer_state(env_seed_gens, policy_gen, shuffle_rng, vec_env)

    def maybe_checkpoint() -> None:
        frac = global_env_steps / trained_env_steps
        # Multiple fraction thresholds can be crossed within one update
        # when updates are coarse relative to the fraction granularity
        # (common for small/smoke runs) — collect ALL newly-crossed
        # fractions first, then write the checkpoint file ONCE for this
        # global_env_steps value, recording every fraction it satisfies.
        newly_crossed = [
            target for target in algo_cfg.checkpoint.fractions if frac + 1e-9 >= target and target not in checkpoints_saved
        ]
        if newly_crossed:
            ckpt_path = run_dir / "checkpoints" / f"step_{global_env_steps}.pt"
            save_checkpoint(
                ckpt_path, agent, global_env_steps, update_idx, resolved_config, h8,
                global_agent_steps=global_agent_steps, extra_rng_state=_trainer_rng_snapshot(),
            )
            checkpoints_saved.update(newly_crossed)
            checkpoint_paths.append(str(ckpt_path))
            manifest.add_checkpoint(
                {"fractions": newly_crossed, "env_steps": global_env_steps, "path": str(ckpt_path)}
            )
        save_checkpoint(
            run_dir / "checkpoints" / "latest.pt", agent, global_env_steps, update_idx, resolved_config, h8,
            global_agent_steps=global_agent_steps, extra_rng_state=_trainer_rng_snapshot(),
        )

    update_idx = start_update - 1
    try:
        if resume_from is None:
            maybe_checkpoint()  # fraction 0.0: the untrained policy
            # Validation point at env step 0, so every learning curve (and the
            # §22.2 AUC) starts from the untrained policy. Evaluation uses its
            # own seeded RNGs — it cannot perturb the training streams.
            step0 = evaluate_on_validation(
                env_cfg, agent.actor, level, n_maps=algo_cfg.eval.n_val_maps, stochastic=algo_cfg.eval.stochastic
            )
            val_logger.log({"update": -1, "env_steps": 0, **step0})

        for update_idx in range(start_update, n_updates):
            buf = RolloutBuffer(
                t_len=t_len, n_envs=e_n, n_agents=n_robots, grid_size=grid_size, state_hw=state_hw
            )

            for _t in range(t_len):
                grid = torch.as_tensor(batch.obs_grid, dtype=torch.float32)
                vec = torch.as_tensor(batch.obs_vec, dtype=torch.float32)
                mask = torch.as_tensor(batch.mask, dtype=torch.bool)
                grid_f = _flatten_en(grid, e_n, n_robots)
                vec_f = _flatten_en(vec, e_n, n_robots)
                mask_f = _flatten_en(mask, e_n, n_robots)
                inputs = _critic_inputs(grid_f, vec_f, batch.state_grid, batch.agent_state, batch.state_vec)

                actions_f, logp_f, value_f = agent.collect(grid_f, vec_f, mask_f, inputs, policy_gen, stochastic=True)
                actions = _unflatten_en(actions_f, e_n, n_robots).numpy().astype(np.int64)
                logp = _unflatten_en(logp_f, e_n, n_robots).numpy()
                value = _unflatten_en(value_f, e_n, n_robots).numpy()

                nxt = vec_env.step(actions)
                reward = nxt.reward.copy()

                if nxt.final is not None:
                    final = nxt.final
                    for local_idx, e in enumerate(final.env_indices):
                        if nxt.truncated[e] and not nxt.terminated[e]:
                            # Bootstrap from the TRUE final state (captured
                            # before auto-reset), for the actor's obs AND,
                            # for MAPPO, the critic's global state (§15.6).
                            final_inputs = _critic_inputs(
                                torch.as_tensor(final.obs_grid[local_idx], dtype=torch.float32),
                                torch.as_tensor(final.obs_vec[local_idx], dtype=torch.float32),
                                final.state_grid[local_idx],
                                final.agent_state[local_idx] if final.agent_state is not None else None,
                                final.state_vec[local_idx],
                            )
                            with torch.no_grad():
                                v_final = agent.critic_value_denormalized(final_inputs)
                            reward[e] += algo_cfg.ppo.gamma * v_final.numpy()

                buf.add(
                    obs_grid=batch.obs_grid.astype(np.uint8),
                    obs_vec=batch.obs_vec,
                    mask=batch.mask,
                    action=actions,
                    logp_old=logp,
                    value_old=value,
                    reward=reward,
                    terminated=nxt.terminated,
                    truncated=nxt.truncated,
                    state_grid=batch.state_grid,
                    state_vec=batch.state_vec,
                    agent_state=batch.agent_state,
                )

                batch = nxt
                global_env_steps += e_n
                global_agent_steps += e_n * n_robots

            grid = torch.as_tensor(batch.obs_grid, dtype=torch.float32)
            vec = torch.as_tensor(batch.obs_vec, dtype=torch.float32)
            grid_f, vec_f = _flatten_en(grid, e_n, n_robots), _flatten_en(vec, e_n, n_robots)
            last_inputs = _critic_inputs(grid_f, vec_f, batch.state_grid, batch.agent_state, batch.state_vec)
            with torch.no_grad():
                last_value = _unflatten_en(agent.critic_value_denormalized(last_inputs), e_n, n_robots)

            buf_t = buf.to_torch()
            advantages, returns = compute_gae(
                buf_t.reward, buf_t.value_old, buf_t.done, last_value, algo_cfg.ppo.gamma, algo_cfg.ppo.gae_lambda
            )

            if agent.value_norm is not None:
                # Statistics from ACTIVE samples only (§16.6); identical to
                # updating on everything when all are active (always in v1).
                agent.value_norm.update(returns if bool(buf_t.active.all()) else returns[buf_t.active])
                returns_n = agent.value_norm.normalize(returns)
            else:
                returns_n = returns
            advantages_n = normalize_advantages(advantages, active=buf_t.active)

            flat = flatten_batch(buf_t, advantages_n, returns_n)
            batch_size = len(flat)
            minibatch_size = max(batch_size // algo_cfg.ppo.minibatches, 1)

            diag_totals: dict[str, float] = {}
            n_mb = 0
            for _epoch in range(algo_cfg.ppo.epochs):
                perm = shuffle_rng.permutation(batch_size)
                for start in range(0, batch_size, minibatch_size):
                    idx = perm[start : start + minibatch_size]
                    idx_t = torch.as_tensor(idx, dtype=torch.long)

                    mb_inputs = CriticInputs(
                        local_grid=flat.obs_grid[idx_t],
                        local_vec=flat.obs_vec[idx_t],
                        state_grid=flat.state_grid[idx_t] if flat.state_grid is not None else None,
                        state_vec=flat.state_vec[idx_t] if flat.state_vec is not None else None,
                    )
                    dist, value_n = agent.evaluate(flat.obs_grid[idx_t], flat.obs_vec[idx_t], flat.mask[idx_t], mb_inputs)
                    actor_loss, critic_loss, diag = compute_update_losses(
                        dist,
                        flat.action[idx_t],
                        flat.logp_old[idx_t],
                        flat.advantages[idx_t],
                        value_n,
                        flat.returns[idx_t],
                        algo_cfg.ppo.entropy_coef,
                        algo_cfg.ppo.clip,
                        active=flat.active[idx_t],
                    )

                    agent.opt_actor.zero_grad()
                    actor_loss.backward()
                    actor_grad_norm = clip_grad_norm(agent.actor, algo_cfg.ppo.max_grad_norm)
                    agent.opt_actor.step()

                    agent.opt_critic.zero_grad()
                    critic_loss.backward()
                    critic_grad_norm = clip_grad_norm(agent.critic, algo_cfg.ppo.max_grad_norm)
                    agent.opt_critic.step()

                    n_mb += 1
                    for k, v in {
                        "policy_loss": float(diag.policy_loss),
                        "value_loss": float(diag.value_loss),
                        "entropy": float(diag.entropy),
                        "approx_kl": float(diag.approx_kl),
                        "clip_fraction": float(diag.clip_fraction),
                        "actor_grad_norm": actor_grad_norm,
                        "critic_grad_norm": critic_grad_norm,
                    }.items():
                        diag_totals[k] = diag_totals.get(k, 0.0) + v

            ev = explained_variance(buf_t.value_old.flatten(), returns.flatten())
            metrics_logger.log(
                {
                    "update": update_idx,
                    "env_steps": global_env_steps,
                    "agent_steps": global_agent_steps,
                    "mean_reward": float(buf_t.reward.mean()),
                    "explained_variance": ev,
                    **{k: v / max(n_mb, 1) for k, v in diag_totals.items()},
                }
            )
            manifest.update_progress(global_env_steps, global_agent_steps, update_idx + 1)

            is_last_update = update_idx == n_updates - 1
            if (update_idx + 1) % algo_cfg.eval.interval_updates == 0 or is_last_update:
                final_val_metrics = evaluate_on_validation(
                    env_cfg, agent.actor, level, n_maps=algo_cfg.eval.n_val_maps, stochastic=algo_cfg.eval.stochastic
                )
                val_logger.log({"update": update_idx, "env_steps": global_env_steps, **final_val_metrics})

            maybe_checkpoint()

        export_actor(
            run_dir / "actor.pt",
            agent,
            meta={"grid_size": grid_size, "hidden": algo_cfg.actor.hidden, "algo": algo_name, "level": level},
        )
        manifest.data["eval_hashes_guarded"] = len(eval_hashes)
        manifest.data["training_maps_rejected_as_eval"] = vec_env.rejected_maps
        manifest.finish("completed", final_val_metrics)
        status = "completed"
    except Exception:
        manifest.finish("failed", final_val_metrics)
        raise
    except BaseException:
        # KeyboardInterrupt / SystemExit (e.g. SIGTERM in a study worker):
        # `latest.pt` from the last finished update makes this resumable.
        manifest.finish("interrupted", final_val_metrics)
        raise
    finally:
        metrics_logger.close()
        val_logger.close()

    return TrainResult(
        run_dir=run_dir,
        global_env_steps=global_env_steps,
        global_agent_steps=global_agent_steps,
        updates=update_idx + 1,
        final_val_metrics=final_val_metrics,
        status=status,
        checkpoints=checkpoint_paths,
    )
