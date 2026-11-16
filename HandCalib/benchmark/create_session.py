"""Create one guide-compliant session folder from the original camera files.

    python create_session.py --collector s01 \
        --before IMG_0001.MOV --after IMG_0010.MOV \
        --twinkle IMG_0002.MOV IMG_0003.MOV --nod IMG_0004.MOV IMG_0005.MOV \
        --circle IMG_0006.MOV IMG_0007.MOV --mix IMG_0008.MOV IMG_0009.MOV \
        [--stabilization off --focus-exposure-locked --shown-on "tablet screen"] [--dry-run]

The session folder name (collector_date_device_lens_resolution[_zoomNN]) and
session.json are filled from the video metadata:

    date         com.apple.quicktime.creationdate (local date), else the camera file name
                 (YYYYMMDD_HHMMSS, e.g. Galaxy), else container creation_time (UTC)
    device       com.apple.quicktime.model
    camera       "back"/"front" in com.apple.quicktime.camera.lens_model
    lens         35 mm equivalent focal length / 25, rounded to a camera-app label
                 (14 mm -> 05x, 24/26 mm -> 1x, 48/52 mm -> 2x, 77 mm -> 3x, 120 mm -> 5x)
    resolution   displayed width x height (rotation applied); token 720p/1080p/4k
    fps          average frame rate, rounded

Fields that the metadata does not hold (stabilization, focus/exposure lock,
checkerboard display, digital zoom, notes) come from options; if an option is
missing the field is left null and listed at the end, and validate_session.py
will report it until it is filled.

Files are copied byte for byte (the guide requires original files; the
camera's metadata, including location, stays in the copy). The copy is
verified by sha256 of the whole file. Source files are never modified.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import qt_meta
from session_rules import MOTIONS, probe

LENS_LABELS = (0.5, 1, 2, 3, 4, 5, 8, 10)
APPLE = "com.apple.quicktime."


def lens_label(f35):
    ratio = f35 / 25.0
    best = min(LENS_LABELS, key=lambda x: abs(ratio / x - 1))
    return best if abs(ratio / best - 1) < 0.15 else None


def lens_token(label):
    return "05x" if label == 0.5 else f"{label:g}x".replace(".", "")


def res_token(w, h):
    short = min(w, h)
    return {2160: "4k", 4320: "8k"}.get(short, f"{short}p")


def describe(path):
    meta = qt_meta.read(path)
    info = probe(path)
    out = {"path": Path(path), "width": info["width"], "height": info["height"],
           "fps": info["fps"], "duration": info["duration"],
           "model": meta.get(APPLE + "model"),
           "lens_model": meta.get(APPLE + "camera.lens_model"),
           "f35": meta.get(APPLE + "camera.focal_length.35mm_equivalent"),
           "cinematic": str(meta.get(APPLE + "cinematic-video-intent", "0")).strip() == "1",
           "date": None}
    created = meta.get(APPLE + "creationdate")
    m = re.match(r"(\d{4})(\d{2})(\d{2})_\d{6}", Path(path).stem)
    if created:
        out["date"] = str(created)[:10]
    elif m:
        out["date"] = f"{m[1]}-{m[2]}-{m[3]}"
    else:
        tags = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time", "-of", "json", str(path)],
            capture_output=True, text=True, check=True).stdout).get("format", {}).get("tags", {})
        if tags.get("creation_time"):
            out["date"] = tags["creation_time"][:10]
            out["date_utc"] = True
    try:
        out["f35"] = float(out["f35"]) if out["f35"] is not None else None
    except ValueError:
        out["f35"] = None
    return out


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def copy_video(src, dst):
    """Byte-for-byte copy of the original file, verified by sha256."""
    if dst.exists():
        raise FileExistsError(dst)
    shutil.copyfile(src, dst)
    h_src, h_dst = file_sha256(src), file_sha256(dst)
    if h_src != h_dst:
        dst.unlink()
        raise RuntimeError(f"{dst.name}: copy differs from source")
    return h_dst


def packet_hash(path):
    """sha256 of the per-packet md5 list of the video stream (container-independent)."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "framemd5", "-"],
                         capture_output=True, text=True, check=True).stdout
    lines = [ln.split(",")[-1].strip() for ln in out.splitlines() if ln and not ln.startswith("#")]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest(), len(lines)


def set_rotation(src, dst, rotation):
    """Re-mux without re-encoding, display rotation set to `rotation` (ffmpeg 4: rotate tag).
    Only the container's display matrix changes; video packets are verified identical."""
    if dst.exists():
        raise FileExistsError(dst)
    subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", str(src), "-map", "0", "-c", "copy", "-map_metadata", "0",
                    "-metadata:s:v:0", f"rotate={int(rotation)}", str(dst)], check=True)
    if packet_hash(src) != packet_hash(dst):
        dst.unlink()
        raise RuntimeError(f"{dst.name}: video packets differ after re-mux")
    return file_sha256(dst), packet_hash(dst)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collector", required=True, help="s01, s02, ...")
    ap.add_argument("--root", default="/public/data/handcalib_data")
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    for m in MOTIONS:
        ap.add_argument(f"--{m}", nargs=2, required=True, metavar=("FILE_01", "FILE_02"))
    ap.add_argument("--stabilization",
                    help='"off", or what was actually used, e.g. "standard (enhanced off)" (only warned about)')
    ap.add_argument("--stabilization-detail",
                    help='what the camera actually does, e.g. "iPhone stock camera: standard EIS + OIS, enhanced off"')
    ap.add_argument("--focus-exposure-locked", action="store_true", default=None)
    ap.add_argument("--shown-on", help='checkerboard display, e.g. "tablet screen" or "printed, rigid board"')
    ap.add_argument("--digital-zoom", type=float, help="1.0 if no digital zoom")
    ap.add_argument("--device", help="override device name when the metadata has none")
    ap.add_argument("--lens", type=float, help="override lens magnification (0.5, 1, 2, 3, 5 ...)")
    ap.add_argument("--camera", choices=("rear", "front"), help="override when the metadata has none")
    ap.add_argument("--notes", help="session notes (lighting, background)")
    ap.add_argument("--display-rotation", type=int, default=None,
                    help="re-mux every file with this display rotation (e.g. 0 when a landscape shot was "
                         "tagged portrait); pixels are not re-encoded, the change is recorded in sources")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, copy nothing")
    args = ap.parse_args()

    if not re.fullmatch(r"s\d{2}", args.collector):
        sys.exit("collector는 s01 같은 형식이어야 합니다")
    roles = {"checkerboard_before": args.before, "checkerboard_after": args.after}
    for m in MOTIONS:
        for i, src in enumerate(getattr(args, m), 1):
            roles[f"hand_{m}_{i:02d}"] = src
    if len(set(Path(p).resolve() for p in roles.values())) != len(roles):
        sys.exit("같은 원본 파일이 두 번 지정되었습니다")

    infos = {}
    for stem, src in roles.items():
        if not Path(src).is_file():
            sys.exit(f"파일 없음: {src}")
        infos[stem] = describe(src)

    problems = []
    for key in ("model", "lens_model", "f35", "width", "height"):
        values = {str(i[key]) for i in infos.values()}
        if len(values) > 1:
            problems.append(f"{key}가 영상마다 다름 ({', '.join(sorted(values))}): 설정이 다른 영상은 같은 세션이 될 수 없습니다")
    fps_values = sorted(i["fps"] for i in infos.values())
    fps_session = fps_values[len(fps_values) // 2]
    for stem, i in infos.items():
        if abs(i["fps"] - fps_session) > 1.0:
            print(f"경고: {i['path'].name}의 평균 fps {i['fps']:.2f}가 다른 영상({fps_session:.2f})보다 낮음. "
                  f"어두워서 프레임이 떨어졌을 수 있음 (가이드 4장: 밝은 곳에서 촬영)")  # fps = frames / played seconds
    for stem, i in infos.items():
        if i["cinematic"]:
            problems.append(f"{stem} ({Path(i['path']).name}): 시네마틱 모드로 촬영됨")
    if problems:
        print("이 파일들로는 세션을 만들 수 없습니다:")
        for p in problems:
            print(f"  - {p}")
        return 1

    ref = infos["checkerboard_before"]
    device = args.device or ref["model"]
    if args.lens is not None:
        lens = args.lens
    elif ref["f35"]:
        lens = lens_label(ref["f35"])
    else:
        lens = None
    camera = args.camera
    if camera is None and ref["lens_model"]:
        lm = ref["lens_model"].lower()
        camera = "front" if "front" in lm else "rear" if "back" in lm else None
    date = ref["date"]
    if ref.get("date_utc"):
        print(f"경고: 날짜를 UTC creation_time에서 읽음 ({date}). 자정 근처 촬영이면 현지 날짜와 다를 수 있음")
    missing_meta = [n for n, v in (("device (--device)", device), ("date", date),
                                    ("lens (--lens)", lens), ("camera (--camera)", camera)) if v is None]
    if ref["f35"] and lens is None and args.lens is None:
        missing_meta.append(f"35mm 환산 {ref['f35']:g}mm가 표준 배율과 맞지 않음. 디지털 줌일 수 있으니 "
                            f"--lens와 --digital-zoom을 직접 지정해야 함")
    if missing_meta:
        print("메타데이터에서 찾지 못한 값이 있어 폴더 이름을 만들 수 없습니다:")
        for m in missing_meta:
            print(f"  - {m}")
        return 1

    w, h = ref["width"], ref["height"]
    if args.display_rotation is not None:
        coded = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                           "stream=width,height", "-of", "json", str(ref["path"])],
                                          capture_output=True, text=True, check=True).stdout)["streams"][0]
        w, h = (coded["width"], coded["height"]) if args.display_rotation % 180 == 0 else (coded["height"], coded["width"])
    zoom = args.digital_zoom
    name = (f"{args.collector}_{date.replace('-', '')}_{re.sub(r'[^a-z0-9]', '', device.lower().replace('+', 'plus'))}_"
            f"{lens_token(lens)}_{res_token(w, h)}")
    if zoom is not None and abs(zoom - 1.0) > 1e-6:
        name += f"_zoom{f'{zoom:g}'.replace('.', '')}"
    session = Path(args.root) / name
    if session.exists():
        print(f"이미 있는 세션 폴더: {session}")
        return 1

    files = {stem: f"{stem}{infos[stem]['path'].suffix.lower()}" for stem in roles}
    meta = {
        "collector": args.collector,
        "date": date,
        "device": device,
        "camera": camera,
        "lens": f"{lens:g}x",
        "digital_zoom": zoom,
        "resolution": [w, h],
        "fps": round(fps_session),
        "stabilization": args.stabilization,
        "stabilization_detail": args.stabilization_detail or "",
        "focus_exposure_locked": args.focus_exposure_locked,
        "checkerboard": {"squares": [8, 8], "inner_corners": [7, 7], "shown_on": args.shown_on,
                         "f_before": None, "f_after": None},
        "reference_f": None,
        "videos": [{"file": files[f"hand_{m}_{i:02d}"], "motion": m, "f": None, "note": ""}
                   for m in MOTIONS for i in (1, 2)],
        "notes": args.notes,
        "camera_metadata": {"lens_model": ref["lens_model"], "focal_length_35mm": ref["f35"]},
        "sources": {files[stem]: {"original": infos[stem]["path"].name} for stem in roles},
    }

    print(f"세션: {session}")
    for stem in roles:
        i = infos[stem]
        print(f"  {files[stem]:<26} <- {i['path'].name}  ({i['duration']:.1f}s)")
    if args.dry_run:
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        print("\n--dry-run: 아무것도 복사하지 않았습니다")
    else:
        session.mkdir(parents=True)
        for stem in roles:
            if args.display_rotation is None:
                digest = copy_video(infos[stem]["path"], session / files[stem])
                meta["sources"][files[stem]]["sha256"] = digest
            else:
                src_rot = probe(infos[stem]["path"])
                digest, vph = set_rotation(infos[stem]["path"], session / files[stem], args.display_rotation)
                meta["sources"][files[stem]].update({
                    "original_sha256": file_sha256(infos[stem]["path"]), "sha256": digest,
                    "display_rotation_changed": {"to": args.display_rotation,
                                                 "displayed_size_before": [src_rot["width"], src_rot["height"]]},
                    "video_packets_sha256": vph, "note": "re-muxed without re-encoding; video packets identical to the original"})
            print(f"  복사 확인: {files[stem]}")
        (session / "session.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")

    todo = []
    if args.stabilization is None:
        todo.append("stabilization: 손떨림 보정을 껐는지 (--stabilization off)")
    if args.focus_exposure_locked is None:
        todo.append("focus_exposure_locked: AE/AF 잠금을 했는지 (--focus-exposure-locked)")
    if args.shown_on is None:
        todo.append('checkerboard.shown_on: 체커보드를 무엇으로 보여줬는지 (--shown-on "tablet screen")')
    if zoom is None:
        todo.append("digital_zoom: 디지털 줌 배율, 쓰지 않았으면 1.0 (--digital-zoom 1.0)")
    if args.notes is None:
        todo.append("notes: 조명, 배경 등 (--notes)")
    if todo:
        print("\n메타데이터에 없어 직접 알려줘야 하는 값 (session.json에 null로 남음):")
        for t in todo:
            print(f"  - {t}")
    print("\n다음 단계:")
    print(f"  python measure_reference.py {session} --write")
    print(f"  python validate_session.py {session}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
