import type { TrainRequestBody, TrainSchema } from "../../api/endpoints/analytics";

export const OVERRIDE_LABELS: Record<string, { label: string; help: string }> = {
  "ppo.lr_actor": { label: "Actor learning rate", help: "Adam step size for the actor" },
  "ppo.lr_critic": { label: "Critic learning rate", help: "Adam step size for the critic" },
  "ppo.entropy_coef": { label: "Entropy coefficient", help: "Exploration bonus weight" },
  "ppo.clip": { label: "PPO clip", help: "Policy ratio clip range" },
  "ppo.gamma": { label: "Discount γ", help: "Reward discount factor" },
  "ppo.gae_lambda": { label: "GAE λ", help: "Advantage estimator bias/variance trade-off" },
  "ppo.epochs": { label: "PPO epochs", help: "Passes over each rollout" },
  "ppo.n_envs": { label: "Parallel envs", help: "Environments stepped per rollout" },
  "ppo.rollout_len": { label: "Rollout length", help: "Steps per env per update" },
  "ppo.minibatches": { label: "Minibatches", help: "Splits of each rollout" },
  "eval.interval_updates": { label: "Validation interval", help: "Updates between validation evaluations" },
};

export interface FormValues {
  level: string;
  algo: string;
  seed: string;
  total_env_steps: string;
  overrides: Record<string, string>; // blank = use the default
}

export type FormErrors = Record<string, string>;

/** Mirrors the backend's `TrainRequest` validator using the bounds the backend itself serves. */
export function validateTrain(v: FormValues, schema: TrainSchema): { errors: FormErrors; body: TrainRequestBody | null } {
  const errors: FormErrors = {};
  const intIn = (raw: string, min: number, max: number, key: string, label: string): number | null => {
    const n = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(n) || !Number.isInteger(n)) errors[key] = `${label} must be a whole number`;
    else if (n < min || n > max) errors[key] = `${label} must be between ${min} and ${max}`;
    else return n;
    return null;
  };
  if (!schema.levels.includes(v.level)) errors.level = "Unknown level";
  if (!schema.algos.includes(v.algo)) errors.algo = "Unknown algorithm";
  const seed = intIn(v.seed, schema.seed.min, schema.seed.max, "seed", "Seed");
  const steps = intIn(v.total_env_steps, schema.total_env_steps.min, schema.total_env_steps.max, "total_env_steps", "Total env steps");
  const overrides: Record<string, number> = {};
  for (const [key, raw] of Object.entries(v.overrides)) {
    if (raw.trim() === "") continue;
    const spec = schema.overrides[key];
    if (!spec) {
      errors[key] = "Not an adjustable parameter";
      continue;
    }
    const label = OVERRIDE_LABELS[key]?.label ?? key;
    const n = Number(raw);
    if (!Number.isFinite(n)) errors[key] = `${label} must be a number`;
    else if (spec.type === "int" && !Number.isInteger(n)) errors[key] = `${label} must be a whole number`;
    else if (n < spec.min || n > spec.max) errors[key] = `${label} must be between ${spec.min} and ${spec.max}`;
    else overrides[key] = n;
  }
  if (Object.keys(errors).length || seed === null || steps === null) return { errors, body: null };
  return { errors, body: { level: v.level, algo: v.algo, seed, total_env_steps: steps, overrides } };
}
