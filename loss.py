# loss.py
import json
import numpy as np

# Load your locked target.json once at import
with open("target.json") as f:
    TARGET = json.load(f)

RADII_TARGET = np.array(TARGET["sholl"]["radii_um"], dtype=float)
COUNTS_TARGET = np.array(TARGET["sholl"]["counts"], dtype=float)

# Scaling factors so different metrics contribute similarly
SCALES = {
    "total_length_um": 2000.0,
    "n_bifurcations": 30.0,
    "n_tips": 34.0,
    "max_tree_depth": 7.0,
    "max_radial_extent_um": 150.0,
    "sholl_l2": 50.0
}

def loss_from_features(sim_features):
    """
    sim_features: dict in the same format as target.json
    Returns a single float loss (lower is better).
    """
    terms = []
    # Scalar feature differences
    for key in ["total_length_um", "n_bifurcations", "n_tips",
                "max_tree_depth", "max_radial_extent_um"]:
        if key in sim_features:
            diff = (sim_features[key] - TARGET[key]) / SCALES[key]
            terms.append(diff**2)

    # Sholl curve L2 difference
    if "sholl" in sim_features:
        sim_r = np.array(sim_features["sholl"]["radii_um"], dtype=float)
        sim_c = np.array(sim_features["sholl"]["counts"], dtype=float)
        sim_interp = np.interp(RADII_TARGET, sim_r, sim_c, left=0, right=0)
        sholl_l2 = np.sqrt(np.mean((sim_interp - COUNTS_TARGET)**2))
        terms.append((sholl_l2 / SCALES["sholl_l2"])**2)

    return float(np.sum(terms))
