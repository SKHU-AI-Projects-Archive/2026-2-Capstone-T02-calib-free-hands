"""Check one session folder, or every session under a data root, against the guide.

    python validate_session.py handcalib_data
    python validate_session.py handcalib_data/s01_20261120_iphone13_1x_1080p

Exit code 1 if any session has an error (a guide violation that requires a retake or a fix).
"""
import argparse
import sys

from session_rules import MANUAL_CHECKS, check_session, find_sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="data root or one session folder")
    ap.add_argument("--no-probe", action="store_true", help="skip ffprobe checks (duration, resolution)")
    args = ap.parse_args()

    failed = 0
    sessions = find_sessions(args.path)
    if not sessions:
        print(f"세션 폴더가 없음: {args.path}")
        return 1
    for session in sessions:
        errors, warnings, _ = check_session(session, probe_videos=not args.no_probe)
        status = "FAIL" if errors else "OK"
        failed += bool(errors)
        print(f"[{status}] {session.name}")
        for e in errors:
            print(f"    오류: {e}")
        for w in warnings:
            print(f"    경고: {w}")
    print(f"\n{len(sessions)}개 세션 중 {len(sessions) - failed}개 통과")
    print("\n눈으로 확인할 항목 (스크립트로 확인 불가):")
    for c in MANUAL_CHECKS:
        print(f"  - {c}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
