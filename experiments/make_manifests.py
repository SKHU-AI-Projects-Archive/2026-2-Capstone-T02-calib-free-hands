"""Generate machine-readable dataset manifests + the human-readable summary.

For every prepared dataset this writes:
    manifests/<name>.json        - summary (counts, bytes, archives, key files)
    manifests/<name>_files.csv   - per-file path / size / mtime / extension

SHA256 is computed for archives and for the small set of important GT files
only; hashing every unpacked file (HanCo alone has >100k) would dominate the
runtime without adding reproducibility value, since the archive hash already
pins the extracted content.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATASETS = HERE / "datasets"
DOWNLOADS = DATASETS / "_downloads"
MANIFESTS = HERE / "manifests"
TODAY = datetime.now(timezone.utc).date().isoformat()


def rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest().upper()


def walk(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p


def file_rows(root: Path):
    rows, total = [], 0
    for p in walk(root):
        st = p.stat()
        total += st.st_size
        rows.append({
            "path": rel(p),
            "size_bytes": st.st_size,
            "modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "extension": p.suffix.lower().lstrip("."),
        })
    return rows, total


def ext_histogram(rows) -> dict:
    hist: dict[str, int] = {}
    for r in rows:
        hist[r["extension"] or "(none)"] = hist.get(r["extension"] or "(none)", 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


def archive_info(paths) -> list[dict]:
    """Hash each archive and verify its listing can be read."""
    out = []
    for p in paths:
        if not p.is_file():
            continue
        info = {"path": rel(p), "size_bytes": p.stat().st_size,
                "sha256": sha256(p), "listing_ok": None, "entries": None,
                "integrity": None}
        try:
            if p.suffix == ".zip":
                import zipfile
                with zipfile.ZipFile(p) as z:
                    names = z.namelist()
                    info["entries"] = len(names)
                    info["listing_ok"] = True
                    bad = z.testzip()
                    info["integrity"] = "ok (CRC verified)" if bad is None else f"CRC failed: {bad}"
            else:
                import tarfile
                with tarfile.open(p) as t:
                    info["entries"] = sum(1 for _ in t)
                    info["listing_ok"] = True
                    info["integrity"] = "ok (listing readable)"
        except Exception as exc:
            info["listing_ok"] = False
            info["integrity"] = f"FAILED: {exc}"
        out.append(info)
    return out


def important(root: Path, patterns, limit: int = 40) -> list[dict]:
    """Locate key GT files by glob pattern and hash them."""
    seen, out = set(), []
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if p.is_file() and p not in seen and len(out) < limit:
                seen.add(p)
                out.append({"path": rel(p), "size_bytes": p.stat().st_size,
                            "sha256": sha256(p), "role": pat})
    return out


def write(name: str, summary: dict, rows: list[dict]) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / f"{name}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        # Datasets with very many small files (HanCo has >100k) get a gzipped
        # file list so the manifest stays a reasonable size in git.
        big = len(rows) > 20000
        out = MANIFESTS / (f"{name}_files.csv.gz" if big else f"{name}_files.csv")
        if big:
            fh = gzip.open(out, "wt", newline="", encoding="utf-8")
        else:
            fh = open(out, "w", newline="", encoding="utf-8")
        with fh as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        summary["files_csv"] = rel(out)
    print(f"  {name}: {len(rows)} files, {summary.get('total_gb')} GB")


def base(dataset, root, rows, total, archives, notes, status="prepared", **kw) -> dict:
    return {
        "dataset": dataset,
        "prepared_date": TODAY,
        "root_path": rel(root) if root else None,
        "file_count": len(rows),
        "total_bytes": total,
        "total_gb": round(total / 1024 ** 3, 4),
        "extension_histogram": ext_histogram(rows),
        "archives": archives,
        "status": status,
        "notes": notes,
        **kw,
    }


def do_gigahands():
    root = DATASETS / "gigahands"
    rows, total = file_rows(root)
    raw = root / "demo_all" / "raw"
    seqs = sorted(p.name for p in (raw / "hand_pose").iterdir() if p.is_dir())
    imp = important(raw, ["hand_pose/*/optim_params.txt"])
    write("gigahands_demo_all", base(
        "GigaHands", root, rows, total,
        archive_info([DOWNLOADS / "gigahands" / "gigahands_demo_all.tar.gz"]),
        ["keypoints_2d are per-view 2D detections with confidence, not curated GT",
         "3D/2D jsonl row index == video frame index; chosen_frames_<hand>.json lists valid 3D frames",
         "camera convention: X_cam = R(qvec wxyz) @ X_world + tvec, OpenCV [k1,k2,p1,p2], metres"],
        components=["hand_pose", "object_pose"],
        subset="demo_all",
        sequences=seqs,
        n_sequences=len(seqs),
        important_files=imp,
        gt={"camera": True, "joints_3d": True, "joints_2d": True, "rgb_video": True},
        ready_for_cam_exp_001=True,
    ), rows)


def do_interhand():
    root = DATASETS / "interhand26m"
    rows, total = file_rows(root)
    ann = root / "annotations"
    sets = {}
    for sub in sorted(p for p in ann.iterdir() if p.is_dir()):
        sets[sub.name] = sorted({f.stem.split("_")[1]
                                 for f in sub.glob("InterHand2.6M_*_data.json")})
    imp = important(ann, ["skeleton.txt", "subject.txt",
                          "human_annot/InterHand2.6M_test_camera.json"])
    write("interhand26m_annotations", base(
        "InterHand2.6M", root, rows, total,
        archive_info([DOWNLOADS / "interhand26m" / "InterHand2.6M.annotations.5.fps.zip"]),
        ["annotations only (5 fps release); no images downloaded",
         "ships NO independent 2D keypoint annotation - official 2D is the projection itself",
         "independent 2D signal for validation is the per-image bbox",
         "camera convention: X_cam = camrot @ (X_world - campos), millimetres, no distortion"],
        components=["annotations"],
        annotation_sets=sets,
        selected_for_cam_exp_001="human_annot/test",
        important_files=imp,
        gt={"camera": True, "joints_3d": True, "joints_2d": False, "rgb_video": False},
        ready_for_cam_exp_001=True,
    ), rows)


def do_hanco():
    root = DATASETS / "hanco"
    rows, total = file_rows(root)
    tester, cm = root / "HanCo_tester", root / "HanCo_calib_meta"
    seqs = sorted(p.name for p in (cm / "calib").iterdir() if p.is_dir()) if (cm / "calib").is_dir() else []
    imp = important(cm, ["meta.json"]) + important(tester, ["meta.json"])
    comp = {}
    for label, d in (("tester", tester), ("calib_meta", cm)):
        if d.is_dir():
            r, t = file_rows(d)
            comp[label] = {"root_path": rel(d), "file_count": len(r),
                           "total_bytes": t, "total_gb": round(t / 1024 ** 3, 4),
                           "top_level": sorted(p.name for p in d.iterdir())}
    tester_seqs = sorted(p.name for p in (tester / "xyz").iterdir()) if (tester / "xyz").is_dir() else []
    write("hanco", base(
        "HanCo", root, rows, total,
        archive_info([DOWNLOADS / "hanco" / "HanCo_tester.zip",
                      DOWNLOADS / "hanco" / "HanCo_calib_meta.zip"]),
        ["two archives kept in separate roots: HanCo_tester (images+xyz+calib for one "
         "sequence) vs HanCo_calib_meta (calib for all sequences + meta.json)",
         "official archive layout preserved (no flattening)",
         "ships NO 2D keypoint annotation; RGB images ARE available, so HanCo is the "
         "best overlay target",
         "camera convention: per-frame K[8] + M[8] 4x4 world->camera, metres, images undistorted",
         f"tester sequences: {tester_seqs}; calib_meta covers {len(seqs)} sequences"],
        components=comp,
        n_calib_sequences=len(seqs),
        tester_sequences=tester_seqs,
        important_files=imp,
        gt={"camera": True, "joints_3d": True, "joints_2d": False, "rgb_video": True},
        ready_for_cam_exp_001=True,
    ), rows)


def do_assemblyhands():
    root = DATASETS / "assemblyhands"
    rows, total = file_rows(root)
    ann = root / "annotations"
    splits = {}
    for sub in sorted(p for p in ann.iterdir() if p.is_dir()):
        splits[sub.name] = sorted(f.name for f in sub.glob("*.json"))
    imp = important(ann, ["skeleton.txt", "demo/*calib*.json",
                          "test-eccv2024/*calib*.json"])
    write("assemblyhands", base(
        "AssemblyHands", root, rows, total,
        archive_info([DOWNLOADS / "assemblyhands" / "drive-download-20260922T044927Z-1-001.zip",
                      DOWNLOADS / "assemblyhands" / "test_HANDS2024_annot.tar"]),
        ["annotation-only download; no RGB / ego images on disk (image overlay not possible)",
         "annotations[].keypoints IS a genuine 2D annotation [u,v,conf]",
         "camera key mismatch: images[].camera = HMC_84358933 vs calib key "
         "HMC_84358933_mono10bit - loader resolves by prefix",
         "camera convention: 3x4 [R|t] per frame per camera, X_cam = R @ X_world + t, millimetres",
         "test-eccv2024 split came from test_HANDS2024_annot.tar, extracted during preparation"],
        components=list(splits),
        splits=splits,
        important_files=imp,
        gt={"camera": True, "joints_3d": True, "joints_2d": True, "rgb_video": False},
        ready_for_cam_exp_001=True,
    ), rows)


def do_reinterhand():
    root = DATASETS / "reinterhand"
    root.mkdir(parents=True, exist_ok=True)
    readme = root / "README.md"
    readme.write_text(
        "# Re:InterHand - not downloaded\n\n"
        f"Status as of {TODAY}: **intentionally not downloaded.**\n\n"
        "The full release is hundreds of GB and is not needed for CAM-EXP-001\n"
        "(GT camera / GT joint loader validation). It is a candidate for the later\n"
        "camera-diversity / generalization stage (CAM-EXP-004 onwards), where a\n"
        "wide range of camera parameters and photorealistic renders is useful.\n\n"
        "When it is fetched, keep the same role separation as the other datasets:\n"
        "- original archives -> `experiments/datasets/_downloads/reinterhand/`\n"
        "- extracted data    -> `experiments/datasets/reinterhand/`\n"
        "- then regenerate the manifest with `python experiments/make_manifests.py`\n",
        encoding="utf-8")
    write("reinterhand", {
        "dataset": "Re:InterHand",
        "prepared_date": TODAY,
        "root_path": rel(root),
        "file_count": 0,
        "total_bytes": 0,
        "total_gb": 0.0,
        "archives": [],
        "components": [],
        "important_files": [],
        "status": "not_downloaded",
        "gt": {"camera": None, "joints_3d": None, "joints_2d": None, "rgb_video": None},
        "ready_for_cam_exp_001": False,
        "notes": ["intentionally not downloaded (hundreds of GB)",
                  "candidate for the later camera-diversity / generalization stage",
                  f"placeholder README written at {rel(readme)}"],
    }, [])


def main() -> None:
    print("generating manifests...")
    do_gigahands()
    do_interhand()
    do_hanco()
    do_assemblyhands()
    do_reinterhand()
    print("done ->", rel(MANIFESTS))


if __name__ == "__main__":
    main()
