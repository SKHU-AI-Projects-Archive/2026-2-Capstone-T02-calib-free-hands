"""Build the nested frame subsets and the pose-diversity subsets.

TARGET-BLIND. No reference focal, no focal error, no fusion outcome takes any
part in any selection here.

Nested temporal subsets: farthest-point insertion in TIME, seeded at the
sequence midpoint. This guarantees N8 subset of N16 subset of N32 subset of
N64 subset of ALL_COMMON, and spreads every subset over the whole duration.

Pose-diversity subsets: the articulation descriptor is the root-relative
reference hand, scale-normalised by its own bone-length sum, with global
rotation removed by Procrustes alignment to the view's medoid pose. DIVERSE is
greedy farthest-point in that space; LOW_DIVERSITY is the N nearest the medoid.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CACHE, FRAME_COUNTS, MANIFESTS, SUM, read_csv, sha256,  # noqa: E402
                    write_csv, write_json)
from compute_hand_profiles import BONES, view_template  # noqa: E402

CAND = MANIFESTS / "cam_exp_008_candidate_frames_v1.csv.gz"
REF_DIR = CACHE / "reference_3d"
OUT = MANIFESTS / "cam_exp_008_frame_subsets_v1.csv.gz"
DIV_N = (16, 32)


def farthest_point_order(values):
    """Deterministic farthest-point insertion order over a 1-D coordinate."""
    v = np.asarray(values, float)
    n = len(v)
    if n == 0:
        return []
    start = int(np.argmin(np.abs(v - np.median(v))))
    chosen = [start]
    d = np.abs(v - v[start])
    for _ in range(1, n):
        nxt = int(np.argmax(d))
        if d[nxt] <= 0:
            remaining = [i for i in range(n) if i not in set(chosen)]
            if not remaining:
                break
            nxt = remaining[0]
        chosen.append(nxt)
        d = np.minimum(d, np.abs(v - v[nxt]))
    return chosen


def procrustes_align(X, Y):
    """Rotate X onto Y (both centred, unit-scaled). Returns rotated X."""
    M = X.T @ Y
    if not np.isfinite(M).all():
        return X
    try:
        U, _, Vt = np.linalg.svd(M)
    except np.linalg.LinAlgError:
        return X
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return X @ R


def pose_descriptors(store, frames, side):
    """Scale-normalised, rotation-removed articulation descriptors."""
    tmpl = view_template(store, frames, side)
    if tmpl is None:
        return [], None
    keys, raw = [], []
    for fr in frames:
        k = f"{fr}|{side}"
        if k + "|xyz" not in store:
            continue
        use = store[k + "|use"]
        if use.sum() < 18:            # need a well-populated hand
            continue
        x = np.asarray(store[k + "|xyz"], float)
        # the descriptor spans all 21 joints, so every joint must be finite -
        # checking only the `use` subset would let NaNs through and break the
        # Procrustes SVD
        if not np.isfinite(x).all():
            continue
        x = x - x[0]
        s = np.linalg.norm(x)
        if not np.isfinite(s) or s <= 0:
            continue
        raw.append(x / s)
        keys.append(fr)
    if len(raw) < 4:
        return [], None
    A = np.stack(raw)
    # medoid before alignment, then align everything to it
    flat = A.reshape(len(A), -1)
    d0 = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    med = int(np.argmin(d0.sum(1)))
    aligned = np.stack([procrustes_align(a, A[med]) for a in A])
    return keys, aligned.reshape(len(aligned), -1)


def main() -> None:
    cand = read_csv(CAND)
    by = defaultdict(list)
    for r in cand:
        by[(r["sequence"], r["camera"])].append(int(r["frame"]))

    rows, meta = [], []
    for (seq, cam), frs in sorted(by.items()):
        frames = sorted(frs)
        order = farthest_point_order(frames)
        nested = {n: sorted(frames[i] for i in order[:n])
                  for n in FRAME_COUNTS if n <= len(frames)}
        for n, sel in nested.items():
            for f in sel:
                rows.append({"sequence": seq, "camera": cam, "subset": f"N{n}",
                             "frame": f})
        for f in frames:
            rows.append({"sequence": seq, "camera": cam,
                         "subset": "ALL_COMMON", "frame": f})

        # pose diversity, pooled over both sides
        ref = REF_DIR / f"{seq}__{cam}.npz"
        div_ok = False
        if ref.exists():
            store = dict(np.load(ref))
            allk, allD = [], []
            for side in ("left", "right"):
                k, D = pose_descriptors(store, frames, side)
                if D is not None:
                    allk += list(k)
                    allD.append(D)
            if allD:
                D = np.vstack(allD)
                kk = np.array(allk)
                # one descriptor per frame: keep the first occurrence
                seen, idx = set(), []
                for i, f in enumerate(kk):
                    if f not in seen:
                        seen.add(f)
                        idx.append(i)
                D, kk = D[idx], kk[idx]
                if len(kk) >= max(DIV_N) :
                    dm = np.linalg.norm(D[:, None, :] - D[None, :, :],
                                        axis=-1)
                    medoid = int(np.argmin(dm.sum(1)))
                    # DIVERSE: greedy farthest point from the medoid
                    chosen = [medoid]
                    dmin = dm[medoid].copy()
                    while len(chosen) < max(DIV_N):
                        nxt = int(np.argmax(dmin))
                        if nxt in chosen:
                            break
                        chosen.append(nxt)
                        dmin = np.minimum(dmin, dm[nxt])
                    near = list(np.argsort(dm[medoid]))
                    for n in DIV_N:
                        if len(chosen) >= n and len(near) >= n:
                            for i in chosen[:n]:
                                rows.append({"sequence": seq, "camera": cam,
                                             "subset": f"DIVERSE{n}",
                                             "frame": int(kk[i])})
                            for i in near[:n]:
                                rows.append({"sequence": seq, "camera": cam,
                                             "subset": f"LOW{n}",
                                             "frame": int(kk[i])})
                            div_ok = True
        meta.append({"sequence": seq, "camera": cam,
                     "n_all_common": len(frames),
                     "has_diversity_subsets": int(div_ok),
                     **{f"n_N{n}": len(v) for n, v in nested.items()}})

    write_csv(OUT, rows)
    write_csv(SUM / "frame_subset_coverage.csv", meta)

    # verify nesting
    bysub = defaultdict(lambda: defaultdict(set))
    for r in rows:
        bysub[(r["sequence"], r["camera"])][r["subset"]].add(int(r["frame"]))
    bad = 0
    for v, d in bysub.items():
        chain = [f"N{n}" for n in FRAME_COUNTS if f"N{n}" in d] + \
                (["ALL_COMMON"] if "ALL_COMMON" in d else [])
        for a, b in zip(chain, chain[1:]):
            if not d[a] <= d[b]:
                bad += 1
    write_json(SUM / "frame_subset_meta.json", {
        "manifest": OUT.name, "sha256": sha256(OUT), "rows": len(rows),
        "views": len(bysub), "nesting_violations": bad,
        "nested_selection": "farthest-point insertion in time, seeded at the "
                            "sequence midpoint",
        "diversity_views": sum(m["has_diversity_subsets"] for m in meta),
        "target_blind": True,
    })
    print(f"subset rows {len(rows)}, views {len(bysub)}, "
          f"nesting violations {bad}, "
          f"diversity views {sum(m['has_diversity_subsets'] for m in meta)}")


if __name__ == "__main__":
    main()
