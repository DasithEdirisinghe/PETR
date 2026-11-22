# RepDETR3D EVA02-L pedestrian cross-assignment audit

Different-candidate cases: 13,521. Threshold: 4 m.

| Measure | Count | Percent |
|---|---:|---:|
| has prior claimed gt within threshold | 7,000 | 51.77% |
| cross assigned center and 2d | 3,363 | 24.87% |
| cross assigned center 2d and 3d | 2,135 | 15.79% |
| alternative better current center and 2d | 7,608 | 56.27% |
| strictly higher alternative 3d iou | 11,346 | 83.91% |
| full assignment cascade | 2,720 | 20.12% |
| strict full assignment cascade | 2,331 | 17.24% |
| cross assigned top10 percent | 2,971 | 21.97% |
| any prior gt closer | 4,534 | 33.53% |
