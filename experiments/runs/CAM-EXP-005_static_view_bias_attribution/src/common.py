"""Shared paths, loading and statistics for CAM-EXP-005.

The observational unit is the (sequence, camera) STATIC VIEW. Frames inside a
view are repeated observations of the same quantity and are never treated as
independent samples. Physical camera id and sequence id are grouping variables
used for clustering and for held-out folds - never as deployable features.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parents[1]
RUNS = RUN_DIR.parent
EXP = RUNS.parent
REPO = EXP.parent
MANIFESTS = EXP / "manifests"

R0031 = RUNS / "CAM-EXP-003_1_distortion_aware_diagnostic"
R004 = RUNS / "CAM-EXP-004_static_camera_multiframe_aggregation"
R0041 = RUNS / "CAM-EXP-004_1_pre_cam005_robustness_audit"

FRAMES64 = MANIFESTS / "gigahands_demo_cam_exp_0041_64frames_v1.csv.gz"
PRED64 = R0041 / "results" / "raw" / "extended_frame_predictions.csv.gz"

RAW = RUN_DIR / "results" / "raw"
SUM = RUN_DIR / "results" / "summary"
TAB = RUN_DIR / "tables"
FIG = RUN_DIR / "figures"
CACHE = RUN_DIR / "cache"

N_BOOT = 10000
SEED = 20260924

PRIMARY_MODEL = "anycalib_gen_radial"
SECONDARY_MODEL = "geocalib_distorted_radial"


# ------------------------------------------------------------------------ io
def read_csv(path) -> list[dict]:
    path = Path(path)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p)


def fnum(x, default=float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------------ targets
def load_view_targets():
    """Per-view signed log focal bias for both frozen models.

        e_v = median_t log( f_pred(v,t) / f_ref(v) )

    f_ref is the dataset-provided reference focal (GT_EFFECTIVE_FOCAL =
    GT_NATIVE_FX in the validated original-image pixel convention, CAM-EXP-002
    focal_usage_audit.md). This is an ANALYSIS TARGET for attribution, not a
    deployment estimator.
    """
    rows = read_csv(PRED64)
    per = {}
    for r in rows:
        if r.get("success") != "1":
            continue
        key = (r["sequence"], r["camera"])
        d = per.setdefault(key, {})
        m = d.setdefault(r["model_key"], {"logs": [], "fx": [], "gt": None})
        gt = fnum(r["gt_fx"])
        fx = fnum(r["pred_fx"])
        if not (np.isfinite(gt) and np.isfinite(fx) and gt > 0 and fx > 0):
            continue
        m["logs"].append(float(np.log(fx / gt)))
        m["fx"].append(fx)
        m["gt"] = gt
    out = []
    for (seq, cam), d in sorted(per.items()):
        row = {"sequence": seq, "camera": cam,
               "physical_camera_id": cam, "n_frames": len(
                   d.get(PRIMARY_MODEL, {}).get("logs", []))}
        for mk, tag in ((PRIMARY_MODEL, "anycalib"),
                        (SECONDARY_MODEL, "geocalib")):
            m = d.get(mk)
            if not m or not m["logs"]:
                continue
            lg = np.asarray(m["logs"])
            row[f"{tag}_signed_log_bias"] = float(np.median(lg))
            row[f"{tag}_abs_log_bias"] = float(abs(np.median(lg)))
            row[f"{tag}_within_view_log_sd"] = float(np.std(lg))
            row[f"{tag}_view_focal_px"] = float(np.median(m["fx"]))
            row[f"{tag}_rel_focal_err_pct"] = float(
                abs(np.median(m["fx"]) - m["gt"]) / m["gt"] * 100)
            row["gt_reference_focal_px"] = m["gt"]
        out.append(row)
    return out


# -------------------------------------------------------------- statistics
def cluster_bootstrap(values, clusters, stat=np.median, n_boot=N_BOOT, seed=SEED):
    """Resample whole clusters with replacement (physical camera by default)."""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    v, c = v[ok], np.asarray(clusters)[ok]
    uniq = np.unique(c)
    members = [np.flatnonzero(c == u) for u in uniq]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[p] for p in pick])
        draws[b] = stat(v[idx])
    return (float(stat(v)), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))


def spearman_cluster_ci(x, y, clusters, n_boot=N_BOOT, seed=SEED):
    """Spearman rho with a cluster bootstrap CI over whole clusters."""
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    c = np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, c = x[ok], y[ok], c[ok]
    if x.size < 8 or np.std(x) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), int(x.size)
    r = stats.spearmanr(x, y)
    uniq = np.unique(c)
    members = [np.flatnonzero(c == u) for u in uniq]
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[p] for p in pick])
        xs, ys = x[idx], y[idx]
        if np.std(xs) == 0 or np.std(ys) == 0:
            continue
        draws[b] = stats.spearmanr(xs, ys).statistic
    d = draws[np.isfinite(draws)]
    lo, hi = ((float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
              if d.size else (float("nan"), float("nan")))
    return float(r.statistic), float(r.pvalue), lo, hi, int(x.size)


def benjamini_hochberg(pvals):
    """Descriptive FDR supplement. Not a headline statistic: the naive p-values
    ignore the camera cluster structure."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return q
    order = idx[np.argsort(p[idx])]
    m = order.size
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = p[i] * m / (rank + 1)
        prev = min(prev, val)
        q[i] = min(prev, 1.0)
    return q
