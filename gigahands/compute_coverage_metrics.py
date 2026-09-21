import sys, json, math
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, '/home/juson/project/hand-demo/gigahands')
import eval_pa_mpjpe_gigahands as baseline
import eval_translation_gigahands as tr

OUT = '/home/juson/project/hand-demo/out/gigahands'
DATA = '/public/data/gigahands'
WRIST = 0

per_inst = pd.read_csv(f'{OUT}/eval_per_instance_gigahands.csv')           # 2804, matched/pa_mpjpe
pose_cache = pd.read_csv(f'{OUT}/gigahands_pose_keypoints_cache.csv')      # 1846 TP, kp3d(21x3)+mano_pose(48)
raw_cache = pd.read_csv(f'{OUT}/gigahands_raw_instances_cache.csv')       # 1846 TP, pred_wrist_gt, gt_wrist_cam, cam=?

canonical = np.load(f'{OUT}/canonical_mano_joints.npy')  # (21,3) right-hand convention, wrist~[0.0957,...]

gt = baseline.load_gt_sequences()
cam_params_cxcy = {}
with open(f'{DATA}/gigahands_optim_params_sample.txt') as f:
    lines = [l.strip() for l in f if l.strip()]
header = lines[0].lstrip('#').split()
for line in lines[1:]:
    row = dict(zip(header, line.split()))
    R = tr.quat_to_rotmat(float(row['qvecw']), float(row['qvecx']), float(row['qvecy']), float(row['qvecz']))
    t = np.array([float(row['tvecx']), float(row['tvecy']), float(row['tvecz'])])
    cam_params_cxcy[row['cam_name']] = dict(fx=float(row['fx']), fy=float(row['fy']),
                                             cx=float(row['cx']), cy=float(row['cy']), R=R, t=t)

def project(cam, p_world):
    p_cam = cam['R'] @ p_world + cam['t']
    u = cam['fx'] * p_cam[0] / p_cam[2] + cam['cx']
    v = cam['fy'] * p_cam[1] / p_cam[2] + cam['cy']
    return np.array([u, v]), p_cam

# ---- GO GT (Rh) 로딩, take 별 캐시 ----
rh_cache = {}
def get_rh(scene, seq_int, side, frame_idx):
    key = (scene, seq_int)
    if key not in rh_cache:
        path = f"{DATA}/gigahands_gt/{scene}/params/{seq_int:03d}.json"
        try:
            with open(path) as f:
                d = json.load(f)
            rh_cache[key] = {s: np.array(d[s]['Rh']) for s in ('left', 'right')}
        except FileNotFoundError:
            rh_cache[key] = None
    entry = rh_cache[key]
    if entry is None or frame_idx >= len(entry[side]):
        return None
    return entry[side][frame_idx]

def geodesic_deg(aa1, aa2):
    R1 = Rotation.from_rotvec(aa1).as_matrix()
    R2 = Rotation.from_rotvec(aa2).as_matrix()
    Rrel = R1.T @ R2
    cos_ang = (np.trace(Rrel) - 1) / 2
    cos_ang = np.clip(cos_ang, -1, 1)
    return math.degrees(math.acos(cos_ang))

MIRROR = np.diag([-1.0, 1.0, 1.0])

def unmirror_global_orient_aa(aa_raw, side):
    """rgb_predictor.py 는 왼손 crop 을 좌우반전해서 넣고 keypoints/vertices 는 x 를 되돌리지만
    mano_pose(global_orient 포함)는 되돌리지 않는다 — 왼손 global_orient 는 '거울에 비친 오른손'
    좌표계로 저장돼 있으므로, 실제 왼손 방향을 얻으려면 R_true = M R_raw M (M=diag(-1,1,1))
    거울 켤레변환을 적용해야 한다. 오른손은 그대로."""
    if side != "left":
        return aa_raw
    R_raw = Rotation.from_rotvec(aa_raw).as_matrix()
    R_true = MIRROR @ R_raw @ MIRROR
    return Rotation.from_matrix(R_true).as_rotvec()

# ---- merge pose_cache + raw_cache (같은 1846 TP, key 동일) ----
key_cols = ['scene', 'seq', 'cam', 'frame', 'side']
pose_idx = pose_cache.set_index(key_cols)
raw_idx = raw_cache.set_index(key_cols)

kp3d_cols = [[f'kp3d_{j}_{a}' for a in 'xyz'] for j in range(21)]
mano_cols = [f'mano_pose_{k}' for k in range(48)]

records = []
n_go_missing = 0
for _, row in per_inst.iterrows():
    scene, seq, cam, frame, side, group, matched = row['scene'], int(row['seq']), row['cam'], int(row['frame']), row['side'], row['group'], row['matched']
    key = (scene, seq, cam, frame, side)
    camp = cam_params_cxcy.get(cam)
    gt_arr = gt[(scene, seq)][side].get(frame)   # (21,3) world
    if camp is None or gt_arr is None:
        continue
    gt_cam_pts = np.array([tr.world_to_cam(camp, p) for p in gt_arr])   # (21,3) camera space, world_to_cam 재사용
    gt_wrist_cam = gt_cam_pts[WRIST]
    # [수정] GT는 world 프레임, WiLoR 예측(kp3d)은 카메라 프레임(X우/Y하/Z전방) — 상대좌표를 그냥
    # 빼면 서로 다른 좌표계끼리 비교하게 된다. 카메라 extrinsic 의 회전만 곱해(병진은 상대좌표라
    # 이미 상쇄) world-frame 상대벡터를 camera-frame 으로 옮긴 뒤 비교한다.
    gt_rel = (camp['R'] @ (gt_arr - gt_arr[WRIST]).T).T   # (21,3) camera-frame 상대좌표 - MPJPE 용

    rh_gt = get_rh(scene, seq, side, frame)
    # GT Rh 도 world-frame 회전이라, 같은 이유로 카메라 회전을 합성해 camera-frame 회전으로 변환
    rh_gt_cam_rotvec = None
    if rh_gt is not None:
        R_gt_cam = camp['R'] @ Rotation.from_rotvec(rh_gt).as_matrix()
        rh_gt_cam_rotvec = Rotation.from_matrix(R_gt_cam).as_rotvec()

    if matched:
        kp3d = pose_idx.loc[key, sum(kp3d_cols, [])].values.astype(np.float64).reshape(21, 3)
        mano_pose = pose_idx.loc[key, mano_cols].values.astype(np.float64)
        pred_wrist_gt = raw_idx.loc[key, ['pred_wrist_gt_x', 'pred_wrist_gt_y', 'pred_wrist_gt_z']].values.astype(np.float64)
        pa_mpjpe_m = row['pa_mpjpe']

        # MPJPE (unaligned, wrist-centered, world/카메라 프레임 관계없이 상대좌표만 비교)
        pred_rel = kp3d - kp3d[WRIST]
        mpjpe_mm = float(np.linalg.norm(pred_rel - gt_rel, axis=1).mean() * 1000.0)

        # EPE: 절대 위치(GT-focal) 21관절 -> 2D 재투영, GT 2D(=GT 3D를 같은 카메라로 투영)와 비교
        pred_abs = pred_wrist_gt + pred_rel   # (21,3) camera space, GT-focal 기준
        epe_list = []
        for j in range(21):
            u_pred, _ = project(camp, pred_abs[j])
            u_gt, _ = project(camp, gt_cam_pts[j])
            epe_list.append(np.linalg.norm(u_pred - u_gt))
        epe_px = float(np.mean(epe_list))

        # GO: 예측 global_orient(axis-angle, mano_pose[:3]) vs GT Rh (왼손은 미러 보정 후)
        go_pred_aa = unmirror_global_orient_aa(mano_pose[:3], side)
        go_deg = geodesic_deg(go_pred_aa, rh_gt_cam_rotvec) if rh_gt_cam_rotvec is not None else None
        if rh_gt is None:
            n_go_missing += 1

        ct_mm = row_ct = None  # CT는 이미 별도 파일에 있음(coverage_aware_ct_and_recall.csv), 여기선 생략
    else:
        can = canonical.copy()
        if side == 'left':
            can[:, 0] = -can[:, 0]
        can_rel = can - can[WRIST]
        mpjpe_mm = float(np.linalg.norm(can_rel - gt_rel, axis=1).mean() * 1000.0)

        can_abs = can_rel  # placeholder translation = 카메라 원점(t=0) 이므로 상대좌표 = 절대좌표
        epe_list = []
        for j in range(21):
            u_pred, _ = project(camp, can_abs[j])
            u_gt, _ = project(camp, gt_cam_pts[j])
            epe_list.append(np.linalg.norm(u_pred - u_gt))
        epe_px = float(np.mean(epe_list))

        pa_err = baseline.pa_mpjpe(can.astype(np.float64), gt_arr.astype(np.float64))  # (21,) per-joint
        pa_mpjpe_m = float(pa_err.mean()) if pa_err is not None else None
        go_deg = geodesic_deg(np.zeros(3), rh_gt_cam_rotvec) if rh_gt_cam_rotvec is not None else None
        if rh_gt is None:
            n_go_missing += 1

    records.append(dict(scene=scene, seq=seq, cam=cam, frame=frame, side=side, group=group,
                         matched=bool(matched), mpjpe_mm=mpjpe_mm, epe_px=epe_px, go_deg=go_deg,
                         pa_mpjpe_mm=(pa_mpjpe_m * 1000.0 if pd.notna(pa_mpjpe_m) else None)))

df = pd.DataFrame(records)
print("총 레코드:", len(df), " (2804 여야 함)")
print("GO GT 없어서 계산 못한 건수:", n_go_missing)
df.to_csv(f'{OUT}/coverage_aware_metrics_gigahands.csv', index=False)
print(f"[저장] {OUT}/coverage_aware_metrics_gigahands.csv")

print("\n=== TP만 (기존 방식) vs -p (TP+FN) ===")
tp = df[df.matched]
print(f"n_TP={len(tp)}  n_total={len(df)}")
print(f"MPJPE (TP만) mean={tp.mpjpe_mm.mean():.2f}mm  median={tp.mpjpe_mm.median():.2f}mm   "
      f"MPJPE-p mean={df.mpjpe_mm.mean():.2f}mm  median={df.mpjpe_mm.median():.2f}mm")
print(f"PA-MPJPE (TP만) mean={tp.pa_mpjpe_mm.mean():.2f}mm  median={tp.pa_mpjpe_mm.median():.2f}mm   "
      f"PA-MPJPE-p mean={df.pa_mpjpe_mm.mean():.2f}mm  median={df.pa_mpjpe_mm.median():.2f}mm  "
      f"(결측 {df.pa_mpjpe_mm.isna().sum()}개)")
print(f"EPE (TP만) mean={tp.epe_px.mean():.2f}px  median={tp.epe_px.median():.2f}px   "
      f"EPE-p mean={df.epe_px.mean():.2f}px  median={df.epe_px.median():.2f}px")
go_tp = tp.go_deg.dropna(); go_all = df.go_deg.dropna()
print(f"GO (TP만, GT있는것) mean={go_tp.mean():.2f}deg n={len(go_tp)}   GO-p mean={go_all.mean():.2f}deg n={len(go_all)}")

print("\n=== GO-p 좌/우 경험적 점검 (컨벤션 문제 있는지) ===")
for side in ('left', 'right'):
    sub = tp[tp.side == side].go_deg.dropna()
    print(f"  {side}: n={len(sub)} mean={sub.mean():.2f}deg median={sub.median():.2f}deg std={sub.std():.2f}deg")
