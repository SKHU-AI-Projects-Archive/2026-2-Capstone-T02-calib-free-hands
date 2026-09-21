"""
PA-MPJPE evaluation of the hand-demo pipeline (AnyHandPredictor, WiLoR backend)
against GigaHands (p005-tea + p005-noodle-fastfood, 154 sequences total).

Does NOT modify any existing hand-demo pipeline code (rgb_predictor.py etc.) —
only imports AnyHandPredictor as a library, same pattern as the earlier
InterHand2.6M eval script.

Sampling strategy (per user request): stratify by camera FOV group. GigaHands'
49-camera rig FOV, converted with the fixed 36mm-equivalent formula, has ZERO
cameras in the 23mm/24mm bins and is dominated by 25/26/27mm — so only those
3 groups are populated, each sampled down to --target-per-group frames (evenly
spread across the sequences that have a downloaded+calibrated camera in that
group), rather than a plain per-sequence subsample.

Metric: PA-MPJPE (Procrustes-aligned MPJPE) computed per detected hand
instance (all 21 joints — GigaHands keypoints_3d has no per-joint validity
mask, unlike InterHand2.6M's valid42), matching hands to GT by AnyHandPredictor's
is_right flag against GigaHands' separate left.jsonl/right.jsonl. A same-side
vs swapped-side diagnostic is also computed to sanity-check that convention.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 eval_pa_mpjpe_gigahands.py --target-per-group 500
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # restrict to GPU 0 only

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent   # 이 스크립트 자신의 위치 (hand-demo/gigahands/)
HAND_DEMO_ROOT = Path("/home/juson/project/hand-demo")

# 원본 GigaHands 데이터셋(11GB+ 영상 포함)은 hand-demo 프로젝트에 넣지 않고 /public/data 에 그대로 둔다
DATA_DIR = Path("/public/data/gigahands")
OUT_DIR = HAND_DEMO_ROOT / "out" / "gigahands"

GT_DIR = DATA_DIR / "gigahands_gt"
VIDEOS_DIR = DATA_DIR / "gigahands_sample_videos"
FOV_CSV = DATA_DIR / "gigahands_fov_result.csv"
MAP_CSV = GT_DIR / "multiview_camera_video_map.csv"

SCENES = ["p005-tea", "p005-noodle-fastfood"]
MM_POINTS = [23, 24, 25, 26, 27]
MM_EDGES = [22.5, 23.5, 24.5, 25.5, 26.5, 27.5]
MIN_VALID_JOINTS = 4


# ---------------------------------------------------------------------------
# FOV group classification (same formula/bin rule as the InterHand2.6M pass)
# ---------------------------------------------------------------------------

def fov_from_mm(f_mm: float) -> float:
    return 2.0 * math.degrees(math.atan(18.0 / f_mm))


def build_bins():
    edges_fov = [fov_from_mm(m) for m in MM_EDGES]
    bins = []
    for i, mm in enumerate(MM_POINTS):
        bins.append((f"{mm}mm", edges_fov[i], edges_fov[i + 1]))  # (label, upper incl, lower excl)
    return bins


def assign_group(fov_deg: float, bins) -> str:
    for label, upper, lower in bins:
        if lower < fov_deg <= upper:
            return label
    return "unclassified"


def load_camera_groups(bins):
    """camera_id -> fov group label, from gigahands_fov_result.csv."""
    cam_group = {}
    with open(FOV_CSV) as f:
        for row in csv.DictReader(f):
            cam_group[row["camera_id"]] = assign_group(float(row["fov_h_deg"]), bins)
    return cam_group


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_downloaded_video_index():
    """basename -> full path, across gigahands_sample_videos/file0..9."""
    index = {}
    for root, _, files in os.walk(VIDEOS_DIR):
        for fn in files:
            if fn.endswith(".mp4"):
                index[fn] = os.path.join(root, fn)
    return index


def load_gt_sequences():
    """
    Returns dict[(scene, seq_int)] -> {
        'left': (frames_dict[frame_idx] = (21,3) array),
        'right': (frames_dict[frame_idx] = (21,3) array),
    }
    Only rows listed in chosen_frames_{side}.json are included (those are the
    curated-trustworthy subset — verified empirically: values are sparse
    indices into the {side}.jsonl array, not a 1:1-length mask).
    """
    gt = {}
    for scene in SCENES:
        seq_root = GT_DIR / scene / "keypoints_3d"
        for seq_name in os.listdir(seq_root):
            seq_dir = seq_root / seq_name
            seq_int = int(seq_name)
            entry = {}
            for side in ("left", "right"):
                jsonl_path = seq_dir / f"{side}.jsonl"
                chosen_path = seq_dir / f"chosen_frames_{side}.json"
                if not jsonl_path.exists() or not chosen_path.exists():
                    entry[side] = {}
                    continue
                with open(jsonl_path) as f:
                    rows = [json.loads(l) for l in f]
                with open(chosen_path) as f:
                    chosen = json.load(f)
                frames = {}
                for idx in chosen:
                    if idx < len(rows):
                        arr = np.array(rows[idx], dtype=np.float64)[:, :3]  # (21,3), drop homogeneous w
                        frames[idx] = arr
                entry[side] = frames
            gt[(scene, seq_int)] = entry
    return gt


def load_video_map():
    """(scene, seq_int) -> {camera_id: relative_video_path_in_csv}"""
    m = {}
    with open(MAP_CSV) as f:
        reader = csv.DictReader(f)
        cam_cols = [c for c in reader.fieldnames if c not in ("scene", "sequence")]
        for row in reader:
            scene = row["scene"]
            if scene not in SCENES:
                continue
            seq_int = int(row["sequence"])
            cams = {c: row[c] for c in cam_cols if row[c]}
            m[(scene, seq_int)] = cams
    return m


# ---------------------------------------------------------------------------
# Candidate building + stratified sampling
# ---------------------------------------------------------------------------

def build_group_candidates(gt, video_map, downloaded_index, cam_group):
    """
    group -> list of (scene, seq_int, camera_id, video_path, sorted_frame_indices)
    where sorted_frame_indices = sorted union(left GT frames, right GT frames)
    for that sequence, and camera_id is the FIRST (alphabetically) available
    (downloaded + calibrated + present-in-map) camera in that group for this
    sequence.
    """
    group_candidates = defaultdict(list)

    for (scene, seq_int), cams in video_map.items():
        if (scene, seq_int) not in gt:
            continue
        entry = gt[(scene, seq_int)]
        frame_union = sorted(set(entry["left"].keys()) | set(entry["right"].keys()))
        if not frame_union:
            continue

        # group available cameras for this sequence by fov group
        by_group = defaultdict(list)
        for cam_id, rel_path in cams.items():
            group = cam_group.get(cam_id)
            if group not in ("25mm", "26mm", "27mm"):
                continue
            basename = rel_path.split("/")[-1]
            if basename not in downloaded_index:
                continue
            by_group[group].append((cam_id, downloaded_index[basename]))

        for group, cam_list in by_group.items():
            cam_list.sort(key=lambda t: t[0])
            cam_id, video_path = cam_list[0]  # deterministic: first by name
            group_candidates[group].append((scene, seq_int, cam_id, video_path, frame_union))

    return group_candidates


def stratified_sample(group_candidates, target_per_group, seed=0):
    """
    For each group, spread target_per_group frames evenly across its candidate
    sequences (evenly-spaced index picks within each sequence's frame_union).
    Returns list of (scene, seq_int, camera_id, video_path, frame_idx, group).
    """
    rng = np.random.default_rng(seed)
    samples = []
    for group in ("25mm", "26mm", "27mm"):
        cands = group_candidates.get(group, [])
        if not cands:
            print(f"  [경고] {group} 그룹에 후보 시퀀스가 없습니다.")
            continue
        cands = sorted(cands, key=lambda c: (c[0], c[1]))
        n = len(cands)
        base = target_per_group // n
        remainder = target_per_group % n
        total_taken = 0
        for i, (scene, seq_int, cam_id, video_path, frame_union) in enumerate(cands):
            quota = base + (1 if i < remainder else 0)
            quota = min(quota, len(frame_union))
            if quota <= 0:
                continue
            if quota >= len(frame_union):
                picks = frame_union
            else:
                pick_idx = np.linspace(0, len(frame_union) - 1, quota).round().astype(int)
                picks = [frame_union[j] for j in sorted(set(pick_idx))]
            for fidx in picks:
                samples.append((scene, seq_int, cam_id, video_path, fidx, group))
            total_taken += len(picks)
        print(f"  {group} 그룹: 후보 시퀀스 {n}개, 목표 {target_per_group}, 실제 확보 {total_taken}")
    return samples


# ---------------------------------------------------------------------------
# PA-MPJPE
# ---------------------------------------------------------------------------

def compute_similarity_transform(S1, S2):
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    var1 = np.sum(X1 ** 2)
    K = X1.dot(X2.T)
    U, _, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    R = V.dot(Z.dot(U.T))
    scale = np.trace(R.dot(K)) / var1
    t = mu2 - scale * (R.dot(mu1))
    return scale * R.dot(S1) + t


def pa_mpjpe(pred_21x3, gt_21x3):
    if pred_21x3.shape[0] < MIN_VALID_JOINTS:
        return None
    p = pred_21x3.T.astype(np.float64)
    g = gt_21x3.T.astype(np.float64)
    p_aligned = compute_similarity_transform(p, g)
    return np.linalg.norm(p_aligned - g, axis=0)  # (21,)


# ---------------------------------------------------------------------------
# Video frame extraction (decode each needed video once, only requested frames)
# ---------------------------------------------------------------------------

def extract_frames(video_path, frame_indices):
    """Sequential decode (no .set() seeking — short clips, avoids seek-accuracy issues)."""
    needed = set(frame_indices)
    if not needed:
        return {}
    max_idx = max(needed)
    out = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return out
    i = 0
    while i <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if i in needed:
            out[i] = frame
        i += 1
    cap.release()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-per-group", type=int, default=500)
    ap.add_argument("--out", type=str, default=str(OUT_DIR / "eval_results_gigahands.json"))
    ap.add_argument("--csv", type=str, default=str(OUT_DIR / "eval_per_instance_gigahands.csv"))
    args = ap.parse_args()

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    bins = build_bins()
    cam_group = load_camera_groups(bins)

    print("=== 데이터 로드 ===")
    downloaded_index = load_downloaded_video_index()
    print(f"  다운로드된 mp4: {len(downloaded_index)}개")
    gt = load_gt_sequences()
    print(f"  GT 시퀀스: {len(gt)}개 (scene 2개 합산)")
    video_map = load_video_map()

    print("\n=== 그룹별 후보 구성 ===")
    group_candidates = build_group_candidates(gt, video_map, downloaded_index, cam_group)
    for g in ("25mm", "26mm", "27mm"):
        print(f"  {g}: 후보 (scene,seq) {len(group_candidates.get(g, []))}개")

    print(f"\n=== 그룹당 목표 {args.target_per_group}프레임 계층 샘플링 ===")
    samples = stratified_sample(group_candidates, args.target_per_group)
    print(f"  총 샘플: {len(samples)}개")

    # group samples by video_path to decode each video only once
    by_video = defaultdict(list)
    for s in samples:
        by_video[s[3]].append(s)

    print(f"\n=== 모델 로드 (backend=wilor, GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}) ===")
    os.chdir(HAND_DEMO_ROOT)  # rgb_predictor's model config uses cwd-relative paths
    sys.path.insert(0, str(HAND_DEMO_ROOT))
    from rgb_predictor import AnyHandPredictor  # noqa: E402
    t0 = time.time()
    predictor = AnyHandPredictor(backend="wilor")
    print(f"  로드 시간: {time.time()-t0:.1f}s")

    records = []
    n_gt_instances = 0
    n_missed = 0
    swap_diag = []  # (same_side_err_mean, swapped_err_mean) for frames with both hands matched+GT

    t_start = time.time()
    n_videos = len(by_video)
    for vi, (video_path, vsamples) in enumerate(by_video.items()):
        frame_indices = [s[4] for s in vsamples]
        frames = extract_frames(video_path, frame_indices)

        for (scene, seq_int, cam_id, _vp, fidx, group) in vsamples:
            entry = gt[(scene, seq_int)]
            gt_left = entry["left"].get(fidx)
            gt_right = entry["right"].get(fidx)
            if gt_left is None and gt_right is None:
                continue
            n_gt_instances += int(gt_left is not None) + int(gt_right is not None)

            frame = frames.get(fidx)
            if frame is None:
                n_missed += int(gt_left is not None) + int(gt_right is not None)
                continue

            try:
                hands = predictor.predict(frame)
            except Exception as e:
                print(f"  [warn] predict 실패 {scene}/{seq_int}/{cam_id} frame{fidx}: {e}")
                n_missed += int(gt_left is not None) + int(gt_right is not None)
                continue

            best_left, best_right = None, None
            for h in hands:
                if h.is_right:
                    if best_right is None or h.score > best_right.score:
                        best_right = h
                else:
                    if best_left is None or h.score > best_left.score:
                        best_left = h

            for side, gt_arr, det in (("left", gt_left, best_left), ("right", gt_right, best_right)):
                if gt_arr is None:
                    continue
                if det is None:
                    n_missed += 1
                    records.append(dict(scene=scene, seq=seq_int, cam=cam_id, frame=fidx,
                                         group=group, side=side, matched=False, pa_mpjpe=None))
                    continue
                err = pa_mpjpe(det.keypoints_3d.astype(np.float64), gt_arr)
                if err is None:
                    n_missed += 1
                    records.append(dict(scene=scene, seq=seq_int, cam=cam_id, frame=fidx,
                                         group=group, side=side, matched=False, pa_mpjpe=None))
                    continue
                records.append(dict(scene=scene, seq=seq_int, cam=cam_id, frame=fidx,
                                     group=group, side=side, matched=True,
                                     pa_mpjpe=float(err.mean())))

            # handedness convention diagnostic (only when both sides present both in GT and detection)
            if gt_left is not None and gt_right is not None and best_left is not None and best_right is not None:
                same_l = pa_mpjpe(best_left.keypoints_3d.astype(np.float64), gt_left)
                same_r = pa_mpjpe(best_right.keypoints_3d.astype(np.float64), gt_right)
                swap_l = pa_mpjpe(best_right.keypoints_3d.astype(np.float64), gt_left)
                swap_r = pa_mpjpe(best_left.keypoints_3d.astype(np.float64), gt_right)
                if same_l is not None and same_r is not None and swap_l is not None and swap_r is not None:
                    swap_diag.append((float((same_l.mean() + same_r.mean()) / 2),
                                       float((swap_l.mean() + swap_r.mean()) / 2)))

        if (vi + 1) % 50 == 0 or (vi + 1) == n_videos:
            elapsed = time.time() - t_start
            print(f"  video {vi+1}/{n_videos} ({elapsed:.0f}s elapsed)")

    # ---- aggregate ----
    matched = [r for r in records if r["matched"]]
    def stats(recs):
        errs = np.array([r["pa_mpjpe"] for r in recs], dtype=np.float64)
        if len(errs) == 0:
            return dict(n=0, mean=None, median=None, std=None)
        return dict(n=len(errs), mean=float(errs.mean()), median=float(np.median(errs)), std=float(errs.std()))

    summary = dict(
        target_per_group=args.target_per_group,
        total_samples=len(samples),
        gt_hand_instances=n_gt_instances,
        matched_instances=len(matched),
        missed_instances=n_gt_instances - len(matched),
        detection_recall=(len(matched) / n_gt_instances) if n_gt_instances else None,
        overall=stats(matched),
        by_group={g: stats([r for r in matched if r["group"] == g]) for g in ("25mm", "26mm", "27mm")},
        by_side={s: stats([r for r in matched if r["side"] == s]) for s in ("left", "right")},
        handedness_diagnostic=dict(
            n_dual_hand_frames=len(swap_diag),
            same_side_mean_mm=float(np.mean([a for a, b in swap_diag])) if swap_diag else None,
            swapped_side_mean_mm=float(np.mean([b for a, b in swap_diag])) if swap_diag else None,
            note="same_side should be much LOWER than swapped_side if is_right<->GT-side convention is correct",
        ),
        units="meters (GigaHands keypoints_3d and WiLoR keypoints_3d are both already in metres; "
              "Procrustes scale absorbs any residual unit mismatch regardless)",
    )

    print("\n=== PA-MPJPE 평가 요약 (GigaHands) ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n요약 저장: {args.out}")

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene", "seq", "cam", "frame", "group", "side", "matched", "pa_mpjpe"])
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"인스턴스별 기록 저장: {args.csv}")


if __name__ == "__main__":
    main()
