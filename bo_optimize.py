import json
import optuna
import numpy as np
from loss import loss_from_features
from simulate import simulate_once

# Keep names consistent with simulate.py and C++ JSON
NAMES = ["speed","persistence","branch_prob","branch_angle_mean_deg",
         "branch_angle_std_deg","prune_prob"]

def objective(trial):
    # Search space (log for small probabilities)
    speed        = trial.suggest_float("speed", 0.2, 2.5)
    persistence  = trial.suggest_float("persistence", 0.2, 0.95)
    branch_prob  = trial.suggest_float("branch_prob", 5e-4, 3e-2, log=True)
    angle_mu     = trial.suggest_float("branch_angle_mean_deg", 10.0, 70.0)
    angle_sd     = trial.suggest_float("branch_angle_std_deg",   5.0, 40.0)
    prune_prob   = trial.suggest_float("prune_prob", 1e-5, 2e-2, log=True)

    params = dict(speed=speed, persistence=persistence, branch_prob=branch_prob,
                  branch_angle_mean_deg=angle_mu, branch_angle_std_deg=angle_sd,
                  prune_prob=prune_prob)

    # Average a few stochastic runs (raise reps for the real sim)
    reps = 3
    losses = []
    for seed in range(reps):
        pf = {**params, "seed": seed}
        feats = simulate_once(pf)  # mock now, real BioDynaMo when BINARY is set
        losses.append(loss_from_features(feats))
    return float(np.mean(losses))

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=20)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=200, show_progress_bar=True)

    best = study.best_params
    best_ordered = {k: best[k] for k in NAMES}  # nice, stable ordering
    print("\nBest params:", json.dumps(best_ordered, indent=2))
    # Save params + a fresh features eval
    feats = simulate_once(best_ordered)
    with open("output/best_params_bo.json","w") as f: json.dump(best_ordered, f, indent=2)
    with open("output/best_features_bo.json","w") as f: json.dump(feats, f, indent=2)
    print("Saved output/best_params_bo.json and output/best_features_bo.json")
