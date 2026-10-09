# SWARMMIND — 8-minute demo script

Everything below uses only the committed real data (`demo_data/`). Start with `make demo` and open
<http://localhost:5173>. Times are for a person talking; `make e2e` runs the same flow in about 40 seconds.

| Min | Page | Do | Say / point out |
|---|---|---|---|
| 0:00 | Swarm Lab, **Replay** | The default replay is IPPO seed 0, final checkpoint. Press **Play** (Space). | A 10×10 world, 3 robots; grey hatching is unexplored. Robots only know what they have sensed (circles). The strip shows time, rescued, coverage, collisions. |
| 1:00 | Swarm Lab | **Policy** → *MAPPO · s0*; **Map** → *bottleneck_vertical*; **Step** → 0 (0%), then 999k (100%). | Same map, same seed: untrained robots wander (✗ 150 steps), trained ones finish in 25. These are recorded replays. |
| 2:00 | Swarm Lab, **Live** | Click **Live**, pick *MAPPO · s0*; open ⚙ Scenario → map source *Structured (held-out)*, **Apply**; map *bottleneck_vertical*; speed 1×, **Play**. | Now the server is running the trained actor step by step. Same seed, same map ⇒ it ends exactly like the stored replay (t=25, 2/2 rescued) — that is checked automatically. Click a robot to see its own 11×11 view. |
| 3:30 | **Algorithm Comparison** | Scroll the table and the per-seed dots. | Random vs IPPO vs MAPPO over 5 seeds each; every number comes from `summary.json`. Both learned policies saturate success, so we compare steps-to-completion: no detectable MAPPO–IPPO difference (that is *not* "they are equal"). |
| 4:30 | **Swarm Analysis** | Heatmaps (switch map/quantity); then **Swarm Evolution** → **Play**. | Where robots go and collide. Four checkpoints of the same run on one clock: 0% → 100% of training. |
| 6:00 | **Train Swarm** | Level L1, IPPO; total env steps 400000; set *Parallel envs* 4, *Rollout length* 64, *PPO epochs* 1; **Start training**; wait for a few points; **Cancel job**. | A real background training process, metrics streamed over a WebSocket. We cancel it — the run is recorded as `interrupted`, not hidden. |
| 7:00 | **Experiment History** | Show the 10 study runs and the cancelled one. | Everything on disk is listed, including unfinished runs. |
| 7:30 | Wrap-up (slide or README) | Read the three findings from the README. | H1 supported; H2 not supported; no claim of emergent coordination. One level, 5 seeds. |

If something misbehaves: the status dot at the bottom-left of the sidebar shows backend health; the Settings page has the API address.
