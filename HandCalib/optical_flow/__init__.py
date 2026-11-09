"""손 광류(KLT) 추적 코드 모음.

klt.py      : 실험 56 이후 주 파이프라인이 쓰는 추적 사슬 (MediaPipe 관절 -> 기준 프레임에서 호모그래피 사슬 -> 추적점)
variants.py : 원인 확인용 실험 변형 (77, 80, 83, 86, 87)
설명과 51-free-shape3d-mmpose 대비 변경 사항은 README.md 에 있다.
"""
from .klt import box_long, chain_points, choose_reference, edge_flags, kt_carry, kt_chain, kt_hand_mask, kt_map, track_hand      # noqa: F401
from .variants import bc_chain, eg_chain, eg_paper, eg_window_frac, kt_chain_log, kv_step, rt_windows, wt_track      # noqa: F401
