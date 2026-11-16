"""Machine-checkable rules from the HandCalib data-collection guide.

The guide (TalkFile_handcalib_guide.html, 2026-11-14) defines a session as
one device + one camera setting with two checkerboard videos and eight hand
videos. Everything a script can verify (names, session.json fields, the
checkerboard 3 % rule, duration, resolution, orientation) is checked here.
Visual rules (one right hand, no blur, constant finger shape, ...) can only
be checked by eye; they are listed in MANUAL_CHECKS and printed as a reminder.
"""
import json
import re
import subprocess
from pathlib import Path

MOTIONS = ("twinkle", "nod", "circle", "mix")
HAND_STEMS = tuple(f"hand_{m}_{i:02d}" for m in MOTIONS for i in (1, 2))
CHECKERBOARD_STEMS = ("checkerboard_before", "checkerboard_after")
VIDEO_EXTS = (".mov", ".mp4")
MIN_SECONDS = 20.0
CHECKERBOARD_MAX_DIFF = 0.03
SESSION_NAME = re.compile(
    r"^(?P<collector>s\d{2})_(?P<date>\d{8})_(?P<device>[a-z0-9]+)_(?P<lens>[0-9]+x)"
    r"_(?P<res>[0-9]+p|[0-9]+k)(?:_zoom(?P<zoom>[0-9]+))?$")
FILE_NAME = re.compile(r"^[a-z0-9_]+\.[a-z0-9]+$")
TOP_KEYS = ("collector", "date", "device", "camera", "lens", "digital_zoom", "resolution",
            "fps", "stabilization", "focus_exposure_locked", "checkerboard", "reference_f",
            "videos", "notes")
CHECKERBOARD_KEYS = ("squares", "inner_corners", "shown_on", "f_before", "f_after")

MANUAL_CHECKS = (
    "오른손 하나만 보이고, 손가락 모양이 처음부터 끝까지 같다.",
    "정면 자세로 시작하고, 손이 선처럼 보이는 순간이 없다.",
    "멈춰서 본 어느 장면에서도 손이 흐리지 않다.",
    "손목부터 손끝까지 한 번도 화면 밖으로 나가지 않았다.",
    "손 길이가 화면 짧은 변의 1/3 이상이다.",
    "영상에 얼굴이나 다른 사람이 찍히지 않았다.",
)


def probe(path):
    """Duration, displayed size (rotation applied) and fps of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    stream = info["streams"][0]
    rotation = 0
    for side in stream.get("side_data_list", []):
        rotation = int(side.get("rotation", rotation))
    rotation = int(stream.get("tags", {}).get("rotate", rotation))
    w, h = stream["width"], stream["height"]
    if rotation % 180:
        w, h = h, w
    duration = float(info["format"]["duration"])
    # Frames per played second. avg_frame_rate divides by the whole media duration, which is
    # wrong when the camera writes an edit list that starts mid-recording (iPhone originals).
    if stream.get("nb_frames") and duration > 0:
        fps = int(stream["nb_frames"]) / duration
    else:
        num, den = (int(x) for x in stream["avg_frame_rate"].split("/"))
        fps = num / den if den else 0.0
    return {"duration": duration, "width": w, "height": h, "fps": fps}


def find_video(session, stem):
    hits = [p for p in session.iterdir() if p.stem == stem and p.suffix.lower() in VIDEO_EXTS]
    return hits[0] if len(hits) == 1 else None


def check_session(session, probe_videos=True):
    """Return (errors, warnings, meta). errors = guide violations that require a retake/fix."""
    session = Path(session)
    errors, warnings = [], []
    meta = {}

    m = SESSION_NAME.match(session.name)
    if not m:
        errors.append(f"세션 폴더 이름이 규칙(수집자_날짜_기기_렌즈_해상도[_zoomNN])과 다름: {session.name}")

    json_path = session / "session.json"
    if not json_path.exists():
        errors.append("session.json 없음")
        return errors, warnings, meta
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"session.json 읽기 실패: {e}")
        return errors, warnings, {}

    for key in TOP_KEYS:
        if key not in meta:
            errors.append(f"session.json 항목 없음: {key}")
        elif meta[key] in (None, "", []) and key != "notes":
            errors.append(f"session.json 항목이 비어 있음: {key}")
    if meta.get("notes", "") == "":
        warnings.append("session.json notes가 비어 있음")
    board = meta.get("checkerboard") or {}
    for key in CHECKERBOARD_KEYS:
        if board.get(key) in (None, "", []):
            errors.append(f"session.json checkerboard.{key} 없음")

    if m:
        if meta.get("collector") != m["collector"]:
            errors.append(f"collector({meta.get('collector')})가 폴더 이름의 수집자 ID({m['collector']})와 다름")
        date = str(meta.get("date", "")).replace("-", "")
        if date != m["date"]:
            errors.append(f"date({meta.get('date')})가 폴더 이름의 날짜({m['date']})와 다름")
    stab = meta.get("stabilization")
    if stab not in (None, "") and str(stab).lower() != "off":
        # The iPhone stock camera cannot turn standard EIS off; such sessions are
        # recorded as they are (e.g. "standard (enhanced off)") and only warned about
        # until the professor decides whether they are acceptable.
        warnings.append(f"stabilization이 off가 아님: {stab} (가이드는 off, 교수님 확인 필요)")
    if meta.get("focus_exposure_locked") is not True:
        errors.append("focus_exposure_locked가 true가 아님")
    if list(board.get("inner_corners", [7, 7])) != [7, 7]:
        warnings.append(f"inner_corners가 calibrator 기본값 (7, 7)과 다름: {board.get('inner_corners')}")

    fb, fa, ref = board.get("f_before"), board.get("f_after"), meta.get("reference_f")
    if all(isinstance(v, (int, float)) for v in (fb, fa)):
        diff = abs(fb - fa) / ((fb + fa) / 2)
        meta["_checkerboard_diff"] = diff
        if diff > CHECKERBOARD_MAX_DIFF:
            errors.append(f"체커보드 두 값 차이 {100 * diff:.1f}% > 3% (세션 전체 재촬영)")
        if isinstance(ref, (int, float)) and abs(ref - (fb + fa) / 2) > 0.5:
            errors.append(f"reference_f({ref})가 f_before/f_after 평균({(fb + fa) / 2:.1f})과 다름")
    if not isinstance(ref, (int, float)) or ref <= 0:
        errors.append("reference_f가 양수 숫자가 아님")

    # files
    for p in sorted(session.iterdir()):
        if p.name == "session.json":
            continue
        if not FILE_NAME.match(p.name):
            errors.append(f"파일 이름에 허용되지 않는 문자: {p.name}")
        elif p.suffix.lower() in VIDEO_EXTS and p.stem not in HAND_STEMS + CHECKERBOARD_STEMS:
            errors.append(f"규칙에 없는 영상 파일: {p.name}")
    videos = {}
    for stem in CHECKERBOARD_STEMS + HAND_STEMS:
        path = find_video(session, stem)
        if path is None:
            errors.append(f"영상 없음 또는 중복: {stem}.(mov|mp4)")
        else:
            videos[stem] = path

    listed = {}
    for v in meta.get("videos") or []:
        listed[v.get("file")] = v
    hand_files = {videos[s].name for s in HAND_STEMS if s in videos}
    for name in sorted(hand_files - set(listed)):
        errors.append(f"session.json videos에 빠진 손 영상: {name}")
    for name in sorted(set(listed) - hand_files):
        errors.append(f"session.json videos에 있는데 폴더에 없는 파일: {name}")
    for name, v in listed.items():
        stem = Path(str(name)).stem
        if stem.startswith("hand_") and v.get("motion") != stem.split("_")[1]:
            errors.append(f"{name}: motion({v.get('motion')})이 파일 이름과 다름")
        if not isinstance(v.get("f"), (int, float)):
            warnings.append(f"{name}: f가 아직 비어 있음 (점검 후 채움)")

    if probe_videos:
        res = meta.get("resolution") or [0, 0]
        for stem, path in videos.items():
            try:
                info = probe(path)
            except (subprocess.CalledProcessError, KeyError, IndexError, ValueError) as e:
                errors.append(f"{path.name}: ffprobe 실패 ({e})")
                continue
            if info["duration"] < MIN_SECONDS:
                errors.append(f"{path.name}: 길이 {info['duration']:.1f}s < 20s")
            if info["width"] < info["height"]:
                errors.append(f"{path.name}: 세로 영상 ({info['width']}x{info['height']})")
            if [info["width"], info["height"]] != list(res):
                errors.append(f"{path.name}: 해상도 {info['width']}x{info['height']}가 "
                              f"session.json resolution {res}과 다름 (원본이 아닐 수 있음)")
            fps = meta.get("fps")
            if isinstance(fps, (int, float)) and abs(info["fps"] - fps) > 1.0:
                warnings.append(f"{path.name}: fps {info['fps']:.2f}가 session.json fps {fps}와 다름")
    return errors, warnings, meta


def find_sessions(root):
    root = Path(root)
    if (root / "session.json").exists():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
