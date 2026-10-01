"""One-off terminology migration applied to the generator sources.

Three wording problems, all of which would have survived regeneration because
they live in the generators:

1. "physical focal" for the 922.77 px figure. Per CAM-EXP-002's own
   focal_usage_audit.md, that number is GT_EFFECTIVE_FOCAL, which the audit
   derives to equal GT_NATIVE_FX exactly - the dataset-provided camera intrinsic
   fx in original full-image pixels. It is not an optical/sensor focal length,
   and the 5000 px figure is a training-convention virtual focal, not a
   measured one.
2. "independent re-run" for E2's 6.46 %. That execution used the same 175
   benchmark views; it is a same-data repeat execution, not independent data.
3. "+/-0.35 pp tolerance" stated as if it were an uncertainty interval. It is
   the observed difference between two full executions, and it must not be
   merged with the 40-frame x 3-repeat spread diagnostic.

After this runs, the generators are re-executed and every artifact is rebuilt.
The script is kept so the change is auditable, not because it needs re-running.
"""
from __future__ import annotations

from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PKG = SCRIPTS.parent

KEY_RENAMES = {
    "cam002_gigahands_physical_focal_median_px":
        "cam002_gt_effective_focal_median_px",
    "cam002_physical_focal_root_error_median_mm":
        "cam002_gt_effective_focal_root_error_median_mm",
    "cam002_physical_focal_absolute_mpjpe_median_mm":
        "cam002_gt_effective_focal_absolute_mpjpe_median_mm",
}

TEXT = [
    # ---- focal terminology -------------------------------------------------
    ("using the dataset-provided physical focal places "
     "the hand far better than the pipeline's assumed "
     "virtual focal",
     "applying the dataset-provided reference focal (GT_EFFECTIVE_FOCAL) in "
     "place of the pipeline's training-convention virtual focal places the hand "
     "far closer to the reference 3D"),
    ('f"physical focal "', 'f"dataset-provided reference focal "'),
    ('"quantity": "virtual vs physical focal, absolute root error"',
     '"quantity": "pipeline virtual focal convention vs dataset-provided "\n'
     '                     "reference focal, absolute root error"'),
    ('"focal length that is far from the physical focal "\n'
     '                               "length of the capture cameras, and this alone "',
     '"focal convention that is far from the dataset-provided "\n'
     '                               "reference focal of the capture cameras in the "\n'
     '                               "same image-coordinate convention, and this alone "'),
    ('"Fig01", "pipeline assumed focal (5000 px) and GigaHands physical focal"',
     '"Fig01", "pipeline virtual focal convention (5000 px) and the "\n'
     '     "dataset-provided reference focal (GT_EFFECTIVE_FOCAL)"'),
    ('{"quantity": "GigaHands physical focal (median)", "value_px": f_phys,',
     '{"quantity": "dataset-provided reference focal, median "\n'
     '                     "(GT_EFFECTIVE_FOCAL = GT_NATIVE_FX)",\n'
     '         "value_px": f_phys,'),
    ('{"quantity": "pipeline assumed focal", "value_px": f_virtual,',
     '{"quantity": "pipeline virtual focal convention "\n'
     '                     "(PIPELINE_BASELINE_FOCAL = FOCAL_LENGTH/IMAGE_SIZE "\n'
     '                     "* max(W,H))", "value_px": f_virtual,'),
    # ---- E2 rerun independence --------------------------------------------
    ('"independent re-run of the identical 8 frames, same environment")',
     '"separate execution on the SAME 175 benchmark views and the "\n'
     '            "identical 8 frames, same environment. A same-data repeat "\n'
     '            "execution, NOT an independent dataset or an independent test.")'),
    ('"test": "leave-one-sequence-out selection and an independent re-run"',
     '"test": "leave-one-sequence-out selection, plus a separate execution on "\n'
     '                 "the same benchmark views"'),
    ('f"independent re-run "', 'f"separate rerun on the same views "'),
    # ---- GeoCalib tolerance ------------------------------------------------
    ('"next_implication": "quote GeoCalib-derived numbers with a +/-0.35 pp "\n'
     '                             "run-to-run tolerance"',
     '"next_implication": "report the observed between-execution difference "\n'
     '                             "(about 0.32 pp on the aggregate median) and "\n'
     '                             "the repeat-diagnostic spreads separately; do "\n'
     '                             "not merge them into one uncertainty interval"'),
    ('"impact": "any GeoCalib-derived number carries a ~+/-0.35 pp run-to-run "\n'
     '                   "tolerance that no bootstrap captures"',
     '"impact": "GeoCalib-derived numbers move between executions. Two full "\n'
     '                   "benchmark executions differed by about 0.32 pp on the "\n'
     '                   "aggregate median; that is an observed difference, not a "\n'
     '                   "statistical uncertainty interval"'),
    ('"selection; GeoCalib numbers with the +/-0.35 pp tolerance"',
     '"selection; GeoCalib numbers reported with the observed "\n'
     '                   "between-execution difference"'),
    ('"carry a ~+/-0.35 pp tolerance."),',
     '"move between executions: the two full benchmark executions run here "\n'
     '         "differed by about 0.32 pp on the aggregate median. Report that "\n'
     '         "observed difference and the 40-frame x 3-repeat spreads "\n'
     '         "separately; they are different measurements and must not be "\n'
     '         "merged into one uncertainty interval."),'),
]


def main() -> None:
    for p in sorted(SCRIPTS.glob("build_*.py")) + [SCRIPTS / "make_figures.py"]:
        s = orig = p.read_text(encoding="utf-8")
        for old, new in KEY_RENAMES.items():
            s = s.replace(old, new)
        for old, new in TEXT:
            s = s.replace(old, new)
        if s != orig:
            p.write_text(s, encoding="utf-8")
            print(f"  patched {p.name}")


if __name__ == "__main__":
    main()
