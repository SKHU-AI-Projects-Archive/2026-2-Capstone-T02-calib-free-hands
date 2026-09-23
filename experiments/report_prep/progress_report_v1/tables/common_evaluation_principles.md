# Common evaluation principles

These rules apply to every experiment in this report and can be quoted directly in Section 3.

| Item | Rule |
|---|---|
| primary statistical unit | one static camera view within one sequence, i.e. (sequence, camera). 175 of them in the camera experiments. |
| frame | a repeated observation of that view, NOT an independent sample. 1400 frames means 175 views x 8 frames. |
| coarser cluster units | physical camera id (40) and sequence (5), used to check that results survive correlated structure. Five clusters is a sensitivity check, not a trustworthy confidence interval. |
| focal reference | the dataset-provided camera intrinsics. |
| provided 3D | a reference / self-consistency target, not an independent external ground truth. |
| provided 2D | dataset-provided 2D observations with confidence, not curated GT. An all-zero (0,0) entry is an observed invalid pattern, not a documented sentinel. |
| model comparison | identical input frames for every model, from a manifest frozen before inference. |
| inference failure | logged with a reason, never silently removed. |
| oracle conditions | anything that uses the ground truth (GT-undistorted images, oracle best-of-N frame choice) is a diagnostic and is never reported as a deployable method. |
| aggregation | never across sequences: a deployment calibrates one video at a time. |
| selection protocol from CAM-EXP-005 | leave-one-sequence-out; selection may only see the four training-fold sequences, and all five held-out numbers are reported. |
| final confirmation | a sealed external dataset, evaluated once, after method development is finished. |
| reproducibility | exact repository commits and checkpoint SHA256 recorded for every external model; GeoCalib is known to vary between runs and its numbers carry a ~+/-0.35 pp tolerance. |
