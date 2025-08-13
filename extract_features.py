#!/usr/bin/env python3
import sys, json
import numpy as np
import morphio

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} input.swc output.json")
    sys.exit(1)

inp_swc, out_json = sys.argv[1], sys.argv[2]

EPS = 1e-6
STEP_UM = 10.0  # matches your target.json radii spacing

def segment_lengths(pts):
    if pts.shape[0] < 2:
        return np.array([], float)
    return np.linalg.norm(pts[1:] - pts[:-1], axis=1)

def collect_sections(nrn, dendrites_only=True):
    def keep(sec):
        return not (dendrites_only and sec.type == morphio.SectionType.axon)
    secs = []
    for root in nrn.root_sections:
        if not keep(root):
            continue
        stack = [root]
        while stack:
            s = stack.pop()
            if not keep(s):
                continue
            secs.append(s)
            stack.extend(s.children)
    return secs

def sholl_curve(sections, soma_center, step_um=STEP_UM):
    max_r = 0.0
    for s in sections:
        pts = np.asarray(s.points, float)
        if pts.size:
            d = np.linalg.norm(pts - soma_center, axis=1)
            if d.size:
                max_r = max(max_r, float(d.max()))
    if max_r <= 0:
        radii = np.arange(step_um, step_um + step_um, step_um, dtype=float)
    else:
        Rmax = float(np.ceil(max_r + step_um))
        radii = np.arange(step_um, Rmax + step_um/2, step_um, dtype=float)

    counts = np.zeros_like(radii, dtype=int)
    for s in sections:
        pts = np.asarray(s.points, float)
        if len(pts) < 2:
            continue
        d = np.linalg.norm(pts - soma_center, axis=1)
        for i in range(len(pts) - 1):
            d1, d2 = d[i], d[i+1]
            s1 = np.sign(d1 - radii)
            s2 = np.sign(d2 - radii)
            sign_change = (s1 * s2) < 0
            c1 = np.isclose(d1, radii, atol=EPS)
            c2 = np.isclose(d2, radii, atol=EPS)
            touches = c1 | c2
            touches_both = c1 & c2
            opposite = (d1 < radii) ^ (d2 < radii)
            counts += (sign_change | (touches & ~touches_both & opposite)).astype(int)
    return radii.tolist(), counts.tolist()

def extract_features(path):
    nrn = morphio.Morphology(path)
    soma_center = np.asarray(nrn.soma.center, float)
    secs = collect_sections(nrn)

    seg_lengths = []
    n_bifurc = n_tips = max_depth = 0
    for root in nrn.root_sections:
        stack = [(root, 0)]
        while stack:
            s, depth = stack.pop()
            pts = np.asarray(s.points, float)
            seg_lengths.extend(segment_lengths(pts))
            max_depth = max(max_depth, depth)
            nchild = len(s.children)
            if nchild == 0:
                n_tips += 1
            if nchild >= 2:
                n_bifurc += 1
            for ch in s.children:
                stack.append((ch, depth + 1))

    seg_lengths = np.asarray(seg_lengths, float)
    total_len = float(seg_lengths.sum()) if seg_lengths.size else 0.0

    max_r = 0.0
    for s in secs:
        pts = np.asarray(s.points, float)
        if pts.size:
            d = np.linalg.norm(pts - soma_center, axis=1)
            if d.size:
                max_r = max(max_r, float(d.max()))

    radii, counts = sholl_curve(secs, soma_center)

    return {
        "total_length_um": total_len,
        "n_bifurcations": int(n_bifurc),
        "n_tips": int(n_tips),
        "max_tree_depth": int(max_depth),
        "max_radial_extent_um": max_r,
        "sholl": {
            "radii_um": radii,
            "counts": counts
        }
    }

features = extract_features(inp_swc)
with open(out_json, "w") as f:
    json.dump(features, f, indent=2)
print(f"Extracted features from {inp_swc} → {out_json}")
