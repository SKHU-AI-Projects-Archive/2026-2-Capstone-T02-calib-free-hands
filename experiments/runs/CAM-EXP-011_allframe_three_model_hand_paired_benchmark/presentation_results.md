# Presentation results

## Main table

| Model | Scene only | Scene + Hand | Gain |
| --- | ---: | ---: | ---: |
| AnyCalib | **16.02 %** | **16.02 %** | +0.00 |
| GeoCalib | **21.35 %** | **21.35 %** | +0.00 |
| Perspective Fields | **19.50 %** | **19.50 %** | +0.00 |

Median relative focal error, **151 paired videos**, every usable RGB frame.

## Frame table

| Sequence | Total frames/video | Usable cameras | Total RGB frames | Hand-usable frames |
| --- | ---: | ---: | ---: | ---: |
| Tea | 381 | 40 | 15,225 | 12,313 |
| Boxing | 366 | 40 | 14,658 | 12,979 |
| Plant | 174 | 40 | 6,962 | 5,197 |
| Dog | 337 | 40 | 13,493 | 11,717 |
| Instrument | 139 | 15 | 2,085 | 2,085 |
| **TOTAL** | | **175** | **52,423** | **44,291** |

"Total frames/video" is the length of the video. "Hand-usable frames" is the
subset where hand geometry could be extracted - it is not a sequence length.

## What to say

> Previously we compared these estimators on 8 frames per video, and a
> follow-up showed the aggregation had largely saturated by 64. Here we used
> every usable RGB frame - 52,423 of them - aggregated each video's frames into
> one focal estimate, and attached the same hand-structure correction to all
> three estimators on the same videos.
>
> The focal error did not change for any of the three.

## Accuracy vs precision (scene only)

| Model | median error | signed bias | within-video spread |
| --- | ---: | ---: | ---: |
| AnyCalib | 16.02 % | +14.66 % | **3.91 %** |
| GeoCalib | 21.35 % | +15.85 % | 14.71 % |
| Perspective Fields | 19.50 % | -11.24 % | 15.81 % |

AnyCalib is very consistent frame-to-frame yet systematically over-estimates;
Perspective Fields is the only one biased downward. These reproduce the earlier
8-frame characterisation closely.

This is a within-model OFF/ON comparison. No model is claimed to be best.
