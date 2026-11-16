"""Measure the reference f of a session from its two checkerboard videos (guide section 3).

    python measure_reference.py handcalib_data/s01_20261120_iphone13_1x_1080p
    python measure_reference.py handcalib_data/s01_... --write   # fill f_before, f_after, reference_f

Uses Calibrator("checkerboard", inner_corners) from calibrator.py unchanged.
"""
import argparse
import json
import sys
from pathlib import Path

from calibrator import Calibrator
from session_rules import CHECKERBOARD_MAX_DIFF, find_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--write", action="store_true", help="write the values into session.json")
    args = ap.parse_args()

    session = Path(args.session)
    json_path = session / "session.json"
    meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    corners = tuple((meta.get("checkerboard") or {}).get("inner_corners") or (7, 7))

    values = {}
    for stem in ("checkerboard_before", "checkerboard_after"):
        path = find_video(session, stem)
        if path is None:
            print(f"{stem}: 영상 없음")
            return 1
        cal = Calibrator("checkerboard", corners)
        try:
            values[stem] = float(cal.run(str(path)))
        except Exception as e:
            print(f"{path.name}: calibrator 오류 ({type(e).__name__}), 보드 검출 {len(cal.observations)}프레임. "
                  f"보드 칸 수(inner_corners={corners})와 촬영 조건을 확인하고 다시 찍어야 합니다.")
            return 1
        print(f"{path.name}: f = {values[stem]:.1f} (보드 검출 {len(cal.observations)}프레임)")

    fb, fa = values["checkerboard_before"], values["checkerboard_after"]
    ref = (fb + fa) / 2
    diff = abs(fb - fa) / ref
    print(f"reference_f = {ref:.1f}, 두 값 차이 {100 * diff:.2f}%")
    if diff > CHECKERBOARD_MAX_DIFF:
        print("두 값이 3% 넘게 다릅니다. 세션 도중 설정이 바뀐 것이므로 처음부터 다시 찍어야 합니다.")

    if args.write:
        meta.setdefault("checkerboard", {})
        meta["checkerboard"]["f_before"] = round(fb, 1)
        meta["checkerboard"]["f_after"] = round(fa, 1)
        meta["reference_f"] = round(ref, 1)
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{json_path}에 저장")
    return 1 if diff > CHECKERBOARD_MAX_DIFF else 0


if __name__ == "__main__":
    sys.exit(main())
