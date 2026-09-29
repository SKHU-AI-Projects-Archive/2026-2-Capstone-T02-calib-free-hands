# Scene model provenance

## The three variants, and why these ones

The CAM-EXP-003 historical primary variants are reused. They were confirmed
against that run's own recorded within-video spreads (AnyCalib 4.6 %, GeoCalib
17.7 %, PF-centered 18.8 %) so the identification is not from memory:

| presentation name | variant | historical key |
| --- | --- | --- |
| AnyCalib | pinhole | `AnyCalib[anycalib_pinhole/pinhole]` |
| GeoCalib | pinhole | `GeoCalib[pinhole]` |
| Perspective Fields | centered | `PerspectiveFields[centered]` |

A better-performing variant was **not** substituted. The adapters in
`experiments/src/calibration/` are reused unchanged, including their checkpoint
loading, input resizing, output focal conversion and failure conventions.

## Scalar focal definition

`sqrt(fx * fy)`, unchanged from CAM-EXP-003 / 005 / 008. The three models
report focals differently (AnyCalib in original-image pixels, GeoCalib as
`camera.f`, Perspective Fields as a relative focal times image height); the
existing validated adapter conversions handle that, and no new definition was
introduced.

## Aggregation

Per video, over that video's valid frames:

```
f_video = exp( median( log f_i ) )
```

the log-domain (geometric) median, matching the CAM-EXP-003/004 convention.

## Failures are recorded, not dropped

A frame where a model returns no usable focal is written with `valid = 0` and a
reason. It is never silently removed, because the failure rate is itself a
result. A video's aggregate requires at least 80 % frame coverage, a rule
frozen before any focal result.

## Two frame sets are reported

| set | definition | role |
| --- | --- | --- |
| `STRICT` | frames where **all three** models returned a valid focal | **primary** |
| `NATIVE` | every frame each model itself succeeded on | secondary |

`STRICT` is primary so that a difference between models cannot come from them
having been scored on different frames.
