"""Render presentation_results_early.md from the frozen evaluation artifacts."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from c012_common import RUN_DIR, SUM, TAB, read_csv, read_json  # noqa

STATUS = ("CAM-EXP-012  PRIMARY PRESENTATION ANALYSIS COMPLETE / "
          "FULL CONTROL ANALYSIS PENDING")


def table(rows, cols=None):
    if not rows:
        return "_(no rows)_\n"
    cols = cols or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main():
    v = read_json(SUM / "cam012_presentation_verdict.json")
    lam = read_csv(TAB / "selected_lambda_by_fold.csv")
    L = []
    A = L.append
    A("# CAM-EXP-012 — presentation results (early, primary only)\n")
    A("```\n%s\n```\n" % STATUS)
    A("The wrong-bone control has not been computed. It is deferred, not\n"
      "cancelled; see `CAM_EXP_012_PRESENTATION_PROTOCOL_AMENDMENT.md`.\n")

    A("\n## Question\n")
    A("Does adding full-video hand-joint geometry evidence to full-video scene\n"
      "focal evidence reduce focal estimation error on unseen physical\n"
      "cameras? The primary comparison is `SCENE_ONLY` against\n"
      "`SCENE_PLUS_CORRECT_HAND`; only the hand term differs.\n")

    A("\n## Protocol\n")
    A("- statistical unit: one sequence-camera video, one vote\n")
    A("- paired set: %d videos scored under every compared condition\n"
      % v["n_paired_videos"])
    A("- hand curves cross-fitted over temporal blocks of 8 frames\n")
    A("- `alpha` from TRAIN videos only, at a fixed +/- 5 percent log-focal offset;\n"
      "  it never sees a reference focal\n")
    A("- `lambda` selected by inner physical-camera CV inside the outer TRAIN\n"
      "  set; `lambda = 0` is not a candidate\n")
    A("- predictions frozen and hashed before the test reference focal was\n"
      "  opened\n")
    A("- uncertainty: paired physical-camera cluster bootstrap, 10,000 iterations\n")

    A("\n## Main result\n")
    A(table(v["main_table"]))

    sup = [r["Model"] for r in v["verdicts"]
           if r["HAND_ON_NUMERICAL_GAIN_SUPPORTED"]]
    A("\n### Answer\n")
    if not sup:
        A("**No.** For all %d scene estimators the hand term left focal error\n"
          "the same or worse on unseen physical cameras.\n"
          "`HAND_ON_NUMERICAL_GAIN_SUPPORTED` is false for every model.\n"
          % len(v["verdicts"]))
        A("\nTwo kinds of column must be read together. `Gain_of_medians_pp`\n"
          "compares the two medians; `Paired_median_gain_pp` and\n"
          "`Paired_mean_gain_pp` are the per-video paired differences, which\n"
          "is what the cluster bootstrap resamples. Where a large share of\n"
          "videos is unchanged the paired median can sit exactly at 0 while\n"
          "the paired mean is clearly negative. That combination means most\n"
          "videos are untouched and the ones that do move mostly move the\n"
          "wrong way.\n")
        nb = sum(1 for r in lam if r["lambda_boundary"] == "1")
        A("\nThe inner camera CV selected the smallest available `lambda`\n"
          "(0.125, the grid's lower boundary) in %d of %d model-by-fold\n"
          "cells. Since `lambda = 0` is excluded by design, that is the\n"
          % (nb, len(lam)) +
          "selection procedure pushing the hand term as close to off as the\n"
          "grid allows. A boundary selection is normally a warning about the\n"
          "grid; here it points the same way as the test result.\n")
    else:
        A("A numerical gain is supported for: %s.\n" % ", ".join(sup))
    A("\n## Controls\n")
    A(table(v["control_table"]))
    A("\n`Wrong_Bone` is `PENDING`. The shuffled control substitutes another\n"
      "video's hand curve through a frozen donor mapping — different sequence,\n"
      "different camera, different participant — reusing the same `alpha` and\n"
      "the same selected `lambda`.\n")

    A("\n## How far the hand term moves the focal\n")
    A(table(v["movement_table"]))

    A("\n## Selected lambda and loss balance\n")
    A(table(lam, ["model", "outer_fold", "D_scene", "D_hand_correct",
                  "before_ratio", "alpha_correct", "after_ratio",
                  "selected_lambda", "lambda_boundary"]))

    A("\n## Verdicts\n")
    A(table(v["verdicts"]))
    A("\n`CORRECT_HAND_SPECIFIC_SIGNAL_SUPPORTED` cannot be evaluated without\n"
      "the wrong-bone control. Even if the correct condition beats both\n"
      "scene-only and the shuffled control, this analysis does not establish\n"
      "that the improvement comes from correct anatomical bone\n"
      "correspondence: the shuffled control rules out only that any hand curve\n"
      "would do, while holding the video fixed and permuting the bones is what\n"
      "isolates correspondence.\n")

    A("\n## Scope\n")
    A("- GigaHands only, and its reference focals vary little across cameras\n"
      "  (`SINGLE_FOCAL_RIG_CONFOUND`), so absolute error levels here are not\n"
      "  transferable to rigs with diverse optics\n")
    A("- the reserved confirmatory holdout remains unopened\n")
    A("- reference 3D geometry is `OTHER_CAMERA_ONLY_REFERENCE_3D`; the target\n"
      "  camera is excluded before reconstruction\n")

    p = RUN_DIR / "presentation_results_early.md"
    p.write_text("\n".join(L), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
