# inspect_bo.py
import json, numpy as np, matplotlib.pyplot as plt

T = json.load(open("target.json"))
S = json.load(open("output/best_features_bo.json"))

def cmp(T,S):
    keys = ["total_length_um","n_bifurcations","n_tips","max_tree_depth","max_radial_extent_um"]
    print("\n== Feature comparison ==")
    for k in keys:
        tv, sv = T[k], S[k]
        print(f"{k:>22}: target={tv:8.3f}  sim={sv:8.3f}  diff={sv-tv:+8.3f}")

def plot_sholl(T,S):
    tr, tc = np.array(T["sholl"]["radii_um"]), np.array(T["sholl"]["counts"])
    sr, sc = np.array(S["sholl"]["radii_um"]), np.array(S["sholl"]["counts"])
    sc_i = np.interp(tr, sr, sc, left=0, right=0)
    plt.figure(figsize=(7,5))
    plt.plot(tr, tc, marker="o", label="Target")
    plt.plot(tr, sc_i, marker="s", label="BO best")
    plt.xlabel("Radius (µm)"); plt.ylabel("Intersections")
    plt.title("Sholl: target vs BO best"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

cmp(T,S)
plot_sholl(T,S)
