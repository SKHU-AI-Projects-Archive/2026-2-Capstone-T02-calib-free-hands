"""외부(협업 파트너) 전달용 폴더 빌더 — hand-extractor 전체 + 가중치·MANO + 샘플 영상을 한 폴더에 모은다.

    python tools/pack_demo.py [--out <폴더>] [--samples <mp4 폴더>] [--no-samples]
                              [--no-weights] [--all-checkpoints]

기본 출력: <repo>/_workspace/hand-demo-package/   (있으면 지우고 다시 만든다)
압축은 하지 않는다 — 필요할 때 이 폴더를 직접 zip 으로 묶으면 된다.

기본 동봉: 소스 전부, models/anyhand_wilor.ckpt + detector.pt + *.yaml, mano_data/MANO_RIGHT.pkl(chumpy 제거본)
          + mano_mean_params.npz, WiLoR/(license.txt 포함), tools/, demo/, samples/*.mp4  (≈ 2.7 GB)
  --all-checkpoints  원본 wilor_final.ckpt(2.4 GB, --checkpoint wilor 비교용)까지
  --no-weights       가중치·MANO 제외 (파트너가 직접 받는 경우 — demo/README.md §2)
항상 제외: 서비스 README.md(앱 연동 문서 — 루트 README 는 demo/README.md 사본), __pycache__, *.pyc,
          mano_data/*.bak(chumpy 원본), .gitignore

⚠️ 라이선스: AnyHand/WiLoR 체크포인트는 CC-BY-NC-ND(비상업·연구 한정), MANO 는 MPI 라이선스상 재배포 금지.
   내부망·연구 협업 범위에서만 전달하고 공개 저장소에 올리지 말 것.
"""
import argparse
import fnmatch
import shutil
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVICE_ROOT.parent.parent
DEFAULT_OUT = REPO_ROOT / "_workspace" / "hand-demo-package"
DEFAULT_SAMPLES = REPO_ROOT / "video" / "Sample"

ALWAYS_EXCLUDE = ["README.md", "mano_data/MANO_RIGHT.pkl.*", "mano_data/*.bak",
                  "**/__pycache__/**", "*.pyc", "**/*.pyc", ".gitignore"]
WEIGHT_GLOBS = ["models/*.ckpt", "models/*.pt", "mano_data/MANO_RIGHT.pkl"]
EXTRA_CHECKPOINTS = ["models/wilor_final.ckpt"]       # --all-checkpoints 일 때만


def _match(rel: str, globs) -> bool:
    return any(fnmatch.fnmatch(rel, g) for g in globs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="출력 폴더 (기존 내용은 지움)")
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="동봉할 mp4 폴더")
    ap.add_argument("--no-samples", action="store_true")
    ap.add_argument("--no-weights", action="store_true", help="가중치·MANO 제외")
    ap.add_argument("--all-checkpoints", action="store_true", help="wilor_final.ckpt 도 동봉")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if out == SERVICE_ROOT or SERVICE_ROOT in out.parents:
        raise SystemExit(f"[error] 출력 폴더가 서비스 폴더 안에 있습니다: {out}")

    files = []   # (src, rel)
    for p in sorted(SERVICE_ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(SERVICE_ROOT).as_posix()
        if _match(rel, ALWAYS_EXCLUDE):
            continue
        if _match(rel, WEIGHT_GLOBS) and (args.no_weights or (rel in EXTRA_CHECKPOINTS and not args.all_checkpoints)):
            continue
        files.append((p, rel))
    files.append((SERVICE_ROOT / "demo" / "README.md", "README.md"))

    if not args.no_samples:
        sdir = Path(args.samples)
        mp4s = sorted(sdir.glob("*.mp4")) if sdir.is_dir() else []
        if not mp4s:
            print(f"[warn] 샘플 영상 없음: {sdir}")
        files += [(m, f"samples/{m.name}") for m in mp4s]

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    total = 0
    for src, rel in files:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        size = src.stat().st_size
        if size > 100e6:
            print(f"    + {rel} ({size / 1e9:.2f} GB)", flush=True)
        shutil.copy2(src, dst)
        total += size
    print(f"[done] {len(files)} files, {total / 1e9:.2f} GB -> {out}")

    weights = [rel for _, rel in files if _match(rel, WEIGHT_GLOBS)]
    if weights:
        print(f"[note] weights bundled ({len(weights)}): research / non-commercial use only (CC-BY-NC-ND, MANO licence)")
    else:
        print("[ok] no weights / MANO in package")
    for _, rel in files:
        if rel.startswith(("models/", "mano_data/", "samples/")):
            print("   ", rel)


if __name__ == "__main__":
    main()
