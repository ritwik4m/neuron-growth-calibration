# optimize.py
import random, json
from loss import loss_from_features
from simulate import simulate_once

# bounds for parameters
BOUNDS = {
    "speed": (0.2, 2.5),
    "persistence": (0.2, 0.95),
    "branch_prob": (0.0005, 0.03),
    "branch_angle_mean_deg": (10.0, 70.0),
    "branch_angle_std_deg": (5.0, 40.0),
    "prune_prob": (0.0, 0.02),
}

def sample_params():
    return {k: (lo + random.random()*(hi-lo)) for k,(lo,hi) in BOUNDS.items()}

best = (1e9, None)
for i in range(60):  # start with 60 trials; increase later
    params = sample_params()
    feats = simulate_once(params)            # mock now; C++ later
    score = loss_from_features(feats)
    if score < best[0]:
        best = (score, {"params": params, "features": feats})
    if (i+1) % 10 == 0:
        print(f"trial {i+1:3d}  best_loss={best[0]:.4f}")

print("\nBEST:")
print(json.dumps(best[1]["params"], indent=2))
print("loss =", best[0])
with open("best_mock.json","w") as f:
    json.dump(best[1], f, indent=2)
print("Saved best_mock.json")
