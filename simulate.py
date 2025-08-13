# simulate.py
import json, os, subprocess, tempfile, shutil

BINARY = "./cpp/neuron_growth"   # change if your binary is elsewhere

def simulate_once(params, reps=1, use_mock_if_missing=True, timeout=180):
    """
    Returns a features dict with the SAME structure as target.json.
    If the BioDynaMo binary isn't found and use_mock_if_missing=True,
    it runs a mock simulator (so you can test the whole pipeline).
    """
    if shutil.which(BINARY) or os.path.exists(BINARY):
        return _simulate_cpp(params, reps=reps, timeout=timeout)
    elif use_mock_if_missing:
        return _simulate_mock(params, reps=reps)
    else:
        raise FileNotFoundError(f"Simulator binary not found at {BINARY}")

def _simulate_cpp(params, reps=1, timeout=180):
    # If your C++ handles stochasticity internally via a seed param, you can
    # run it multiple times and average. For now, call once.
    with tempfile.TemporaryDirectory() as td:
        pjson = os.path.join(td, "params.json")
        fjson = os.path.join(td, "features.json")
        with open(pjson, "w") as f:
            json.dump(params, f)
        # Convention: ./cpp/neuron_growth <params.json> <features.json>
        subprocess.run([BINARY, pjson, fjson], check=True, timeout=timeout)
        with open(fjson) as f:
            return json.load(f)

# ---------- MOCK SIMULATOR (simple, deterministic) ----------
# This gives you something to optimize against *now*.
def _simulate_mock(params, reps=1):
    import math, numpy as np

    # Params you’ll later mirror in C++:
    speed           = float(params.get("speed", 1.0))          # µm/step
    persistence     = float(params.get("persistence", 0.6))    # 0..1
    branch_prob     = float(params.get("branch_prob", 0.01))   # per step
    angle_mu        = float(params.get("branch_angle_mean_deg", 30.0))
    angle_sd        = float(params.get("branch_angle_std_deg", 15.0))
    prune_prob      = float(params.get("prune_prob", 0.002))

    # Heuristic mapping → features (rough shape so the loss responds sensibly)
    # Target ballpark (your neuron):
    T_LEN   = 1843.0
    T_BIF   = 30.0
    T_TIPS  = 34.0
    T_DEPTH = 7.0
    T_RAD   = 152.0

    # crude relationships (you’ll replace with real simulation output)
    growth_factor = speed * (0.6 + 0.8*persistence) * (1.0 - 5.0*prune_prob)
    branching_factor = 900.0 * branch_prob * (1.0 + 0.01*max(0, angle_sd - 10))
    spread_factor = 80.0 + 120.0 * persistence * (1.0 + 0.01*(angle_mu-30))

    total_length_um = max(100.0, growth_factor * 1500.0)
    n_bifurcations  = max(1, int(branching_factor))
    n_tips          = max(1, int(0.8*n_bifurcations + 8))
    max_tree_depth  = max(1, int(3 + 6*persistence - 0.02*angle_sd))
    max_radial_extent_um = max(30.0, spread_factor)

    # mock Sholl curve: lognormal-ish bell with peak ~spread_factor/4
    radii = list(range(10, 160, 10))
    peak_r = max(20.0, min(60.0, spread_factor/3.0))
    peak_h = 10 + int(0.015*total_length_um)  # higher when length is larger
    counts = []
    for r in radii:
        x = (r - peak_r) / 15.0
        counts.append(max(0, int(peak_h * math.exp(-0.5*x*x))))
    # light prune effect on outer rings
    counts = [max(0, int(c * (1.0 - 2.0*prune_prob * (i/len(radii))))) for i, c in enumerate(counts)]

    return {
        "total_length_um": total_length_um,
        "n_bifurcations": n_bifurcations,
        "n_tips": n_tips,
        "max_tree_depth": max_tree_depth,
        "max_radial_extent_um": max_radial_extent_um,
        "sholl": {"radii_um": radii, "counts": counts}
    }
