# inspect_best.py
import json, numpy as np, matplotlib.pyplot as plt

with open("target.json") as f:
    T = json.load(f)
with open("best_mock.json") as f:
    B = json.load(f)
S = B["features"]

def show_table(T, S):
    keys = ["total_length_um","n_bifurcations","n_tips",
            "max_tree_depth","max_radial_extent_um"]
    print("\n== Feature comparison (target vs best) ==")
    for k in keys:
        tv, sv = T[k], S[k]
        print(f"{k:>22}:  target={tv:8.3f}   sim={sv:8.3f}   diff={sv-tv:+8.3f}")

def plot_sholl(T, S):
    tr, tc = np.array(T["sholl"]["radii_um"]), np.array(T["sholl"]["counts"])
    sr, sc = np.array(S["sholl"]["radii_um"]), np.array(S["sholl"]["counts"])

    # align via interpolation (same as loss)
    sc_interp = np.interp(tr, sr, sc, left=0, right=0)

    plt.figure(figsize=(7,5))
    plt.plot(tr, tc, marker="o", label="Target")
    plt.plot(tr, sc_interp, marker="s", label="Sim (best)")
    plt.xlabel("Radius from soma (µm)"); plt.ylabel("Intersections")
    plt.title("Sholl: target vs best")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

show_table(T, S)
plot_sholl(T, S)
