# Reproduction of CAM-EXP-009.3

## Source run

| field | value |
| --- | --- |
| run | `experiments\runs\CAM-EXP-009_3_real_bilateral_geometry_separability` |
| last commit touching it | `1b92dc1` |
| working tree clean | True |
| reference reconstruction re-run | **No** |

The 14,015 `loco.reconstruct` calls were **not** repeated. CAM-EXP-009.3's
stored frame bone vectors and geometry templates are the source of truth, reused
byte-identically. Every reused file's path, size, SHA-256 and row count is in
`tables/source_artifact_hashes.csv`.

## Recorded facts, re-checked against the files

Each of CAM-EXP-009.3's recorded facts was verified against the actual
artefacts rather than trusted from the report:

| fact | expected | actual | |
| --- | ---: | ---: | --- |
| reconstructions attempted | 14,015 | 14,015 | OK |
| usable frame bone vectors | 13,903 | 13,903 | OK |
| `target_camera_in_reference_count` | 0 | 0 | OK |
| bones in the representation | 20 | 20 | OK |
| absolute scale removed (`sum p = 1`) | yes | yes | OK |
| max frames per unit | ≤ 48 | 48 | OK |

Had any of these disagreed, the file's value would have been reported and the
reanalysis stopped rather than the remembered number being forced.

## v1 headline metrics, recomputed

All recomputed from the stored distance tables at tolerance 1e-9:

| metric | expected | recomputed | pass |
| --- | ---: | ---: | --- |
| D_repeat | 0.0108707811115114 | 0.0108707811115114 | PASS |
| D_view_repeat | 0.0113009022650659 | 0.0113009022650659 | PASS |
| D_within | 0.0282863571345993 | 0.0282863571345993 | PASS |
| D_cross | 0.0457935096030396 | 0.0457935096030396 | PASS |
| ratio within/cross | 0.617693585396689 | 0.617693585396689 | PASS |
| S_sep | 1.61047787540322 | 1.61047787540322 | PASS |
| frac M_nearest > 0 | 0.536842105263158 | 0.536842105263158 | PASS |
| top-1 combined | 53.6842105263158 | 53.6842105263158 | PASS |
| C0 median d_within | 0.0282863571345993 | 0.0282863571345993 | PASS |
| C1 median d_within | 0.270646594049933 | 0.270646594049933 | PASS |
| C1 degradation % | 856.809647711348 | 856.809647711348 | PASS |

**ALL_REPRODUCED = true.** `tables/cam0093_v1_reproduction_audit.csv`.

## Status of CAM-EXP-009.3

CAM-EXP-009.3 **remains valid** as a within-sequence vs cross-sequence analysis,
and its numbers stand unchanged. Its **participant-provenance interpretation**
is superseded by this reanalysis: the local-file audit genuinely found no
participant field, but the convention is documented upstream.

CAM-EXP-009.3 was not an error of analysis. It was a correct conclusion from an
incomplete provenance search, and its historical verdict is left on the record
untouched.
