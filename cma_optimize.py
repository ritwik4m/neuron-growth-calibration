# cma_optimize.py
import json, cma, numpy as np
from loss import loss_from_features
from simulate import simulate_once

# same parameter order everywhere:
NAMES = ["speed","persistence","branch_prob","branch_angle_mean_deg","branch_angle_std_deg","prune_prob"]
LB = np.array([0.2, 0.2, 0.0005, 10.0,  5.0, 0.0])
UB = np.array([2.5, 0.95, 0.03,   70.0, 40.0, 0.02])

x0 = np.array([1.2, 0.6, 0.01, 40.0, 20.0, 0.004])  # a reasonable center
sigma0 = 0.3

def clamp(x): return np.minimum(UB, np.maximum(LB, x))

def f(x):
    p = {k: float(v) for k, v in zip(NAMES, clamp(x))}
    feats = simulate_once(p)  # mock for now, real C++ later
    return loss_from_features(feats)

es = cma.CMAEvolutionStrategy(x0, sigma0, {"bounds":[LB.tolist(), UB.tolist()], "popsize":20})
while not es.stop():
    xs = es.ask()
    ys = [f(x) for x in xs]
    es.tell(xs, ys)
    es.disp()

best_x = es.result.xbest
best_p = {k: float(v) for k, v in zip(NAMES, clamp(best_x))}
print("BEST PARAMS:", json.dumps(best_p, indent=2))
with open("best_mock.json","w") as f:
    json.dump({"params": best_p, "features": simulate_once(best_p)}, f, indent=2)
