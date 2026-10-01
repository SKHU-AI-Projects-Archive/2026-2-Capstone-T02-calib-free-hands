"""Gate: is the p<NN> prefix an OFFICIAL participant identifier?

CAM-EXP-009.3 searched only the locally shipped data files and found no
participant field, so it refused to use the prefix. This checks the official
GigaHands repository README, which documents the dataset's directory layout.

The identity source is the official naming convention and nothing else. MANO
`shapes` similarity is explicitly NOT used: defining identity from fitted hand
shape and then measuring hand-shape similarity would be circular.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MANIFESTS, PARTICIPANT_OF, RUN_DIR, SUM, TAB,  # noqa: E402
                    sha256, write_csv, write_json)

SNAPSHOT = RUN_DIR / "notes" / "gigahands_README_snapshot.md"
SOURCE_URL = "https://github.com/Kristen-Z/GigaHands"
RAW_URL = "https://raw.githubusercontent.com/Kristen-Z/GigaHands/main/README.md"
ACCESS_DATE = "2026-09-28"

PATTERN = re.compile(r"p<participant id>-<scene>")


def main():
    if not SNAPSHOT.exists():
        write_json(SUM / "identity_provenance_summary.json",
                   {"STATUS": "OFFICIAL_PARTICIPANT_MAPPING_NOT_VERIFIED",
                    "reason": "official README snapshot not available"})
        raise SystemExit("OFFICIAL_PARTICIPANT_MAPPING_NOT_VERIFIED")

    text = SNAPSHOT.read_text(encoding="utf-8", errors="ignore")
    # keep only the documented path token, not the tree-drawing characters
    quotes = []
    for ln in text.splitlines():
        m = re.search(r"p<participant id>-<scene>[^\s]*", ln)
        if m and m.group(0) not in quotes:
            quotes.append(m.group(0))
    subjects = [ln.strip() for ln in text.splitlines()
                if "56 subjects" in ln]

    verified = bool(quotes)
    status = ("OFFICIAL_PARTICIPANT_MAPPING_VERIFIED" if verified
              else "OFFICIAL_PARTICIPANT_MAPPING_NOT_VERIFIED")

    rows = []
    for seq, pid in sorted(PARTICIPANT_OF.items()):
        prefix, rest = seq.split("-", 1)
        scene = rest.rsplit("-", 1)[0]
        seqid = rest.rsplit("-", 1)[1] if "-" in rest else ""
        rows.append({
            "sequence": seq, "participant_id": pid,
            "scene": scene, "sequence_id": seqid,
            "identity_source": "OFFICIAL_GIGAHANDS_README_NAMING_CONVENTION",
            "identity_confidence": "VERIFIED" if verified else "UNVERIFIED",
            "same_participant_comparison_allowed": "YES" if verified else "NO",
        })
    write_csv(TAB / "official_participant_mapping.csv", rows)
    mp = MANIFESTS / "cam_exp_00931_participant_mapping_v1.csv"
    write_csv(mp, rows)

    out = {
        "STATUS": status,
        "official_source_title": "GigaHands official repository README",
        "official_source_url": SOURCE_URL,
        "raw_url": RAW_URL,
        "access_date": ACCESS_DATE,
        "snapshot_path": str(SNAPSHOT.relative_to(RUN_DIR)),
        "snapshot_sha256": sha256(SNAPSHOT),
        "verbatim_directory_lines": quotes,
        "verbatim_scale_line_fragment":
            ("56 subjects" if subjects else None),
        "why_this_is_not_filename_intuition":
            "The README's own directory specification names the field: it "
            "writes the take directory as `p<participant id>-<scene>-<squence "
            "id>/`. The token `p` is therefore a dataset-defined identifier "
            "prefix whose semantics the dataset documents, not a pattern "
            "guessed from a filename. The local layout "
            "`demo_all/raw/hand_pose/p36-tea-0010/` matches that "
            "specification exactly.",
        "identity_source_excludes": [
            "MANO shapes similarity (circular: defines identity from hand "
            "shape, then measures hand-shape similarity)",
            "filename intuition",
            "any CAM-EXP-009.3 assumption",
        ],
        "mapping": PARTICIPANT_OF,
        "n_sequences": len(PARTICIPANT_OF),
        "n_participants": len(set(PARTICIPANT_OF.values())),
        "multi_session_participants": {
            p: [s for s, q in PARTICIPANT_OF.items() if q == p]
            for p in sorted(set(PARTICIPANT_OF.values()))
            if sum(1 for q in PARTICIPANT_OF.values() if q == p) > 1},
        "supersedes": "CAM-EXP-009.3's SAME_SUBJECT_PROVENANCE_UNRESOLVED, "
                      "which was the correct outcome of a LOCAL-FILE-ONLY "
                      "audit. The local files genuinely contain no participant "
                      "field; the convention is documented upstream.",
        "power_note": "Verifying identity does not fix statistical power. "
                      "4 participants remains below the pre-set bar of 8.",
    }
    write_json(SUM / "identity_provenance_summary.json", out)

    print("STATUS:", status)
    for q in quotes:
        print("  quote:", q)
    print("  participants:", out["n_participants"],
          "multi-session:", out["multi_session_participants"])
    if not verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
