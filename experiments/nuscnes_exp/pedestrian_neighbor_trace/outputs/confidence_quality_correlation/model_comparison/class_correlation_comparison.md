# PETR versus StreamPETR confidence–3D-IoU correlation

Each prediction is paired with its best same-class GT 3D IoU in the same validation frame. Both models use the same nuScenes maximum-box and class-range filters. No TP matching or center-distance threshold is used. The positive subset contains only predictions with best 3D IoU > 0.

| Class | PETR n | StreamPETR n | PETR positive % | StreamPETR positive % | PETR Pearson all | StreamPETR Pearson all | PETR Spearman all | StreamPETR Spearman all | PETR Pearson IoU>0 | StreamPETR Pearson IoU>0 | PETR Spearman IoU>0 | StreamPETR Spearman IoU>0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Car | 476,300 | 495,252 | 21.3 | 26.9 | 0.748 | 0.714 | 0.502 | 0.471 | 0.626 | 0.624 | 0.564 | 0.542 |
| Truck | 155,308 | 124,458 | 15.2 | 21.6 | 0.636 | 0.644 | 0.402 | 0.440 | 0.567 | 0.571 | 0.484 | 0.503 |
| Bus | 22,146 | 18,109 | 26.2 | 39.2 | 0.716 | 0.639 | 0.523 | 0.502 | 0.624 | 0.558 | 0.533 | 0.465 |
| Trailer | 30,586 | 24,108 | 24.7 | 30.9 | 0.478 | 0.486 | 0.406 | 0.391 | 0.359 | 0.343 | 0.268 | 0.306 |
| Construction Vehicle | 29,374 | 26,342 | 11.2 | 14.1 | 0.469 | 0.534 | 0.287 | 0.322 | 0.442 | 0.533 | 0.310 | 0.373 |
| Pedestrian | 350,034 | 298,631 | 5.8 | 10.3 | 0.441 | 0.456 | 0.304 | 0.342 | 0.167 | 0.213 | 0.155 | 0.180 |
| Motorcycle | 35,886 | 31,608 | 6.7 | 12.2 | 0.529 | 0.555 | 0.304 | 0.340 | 0.307 | 0.366 | 0.290 | 0.274 |
| Bicycle | 55,598 | 42,618 | 3.6 | 7.5 | 0.466 | 0.519 | 0.233 | 0.282 | 0.243 | 0.317 | 0.219 | 0.227 |
| Traffic Cone | 82,227 | 79,653 | 8.1 | 14.4 | 0.408 | 0.519 | 0.342 | 0.411 | 0.209 | 0.321 | 0.216 | 0.304 |
| Barrier | 48,809 | 58,726 | 37.0 | 45.5 | 0.448 | 0.521 | 0.519 | 0.458 | 0.209 | 0.392 | 0.221 | 0.341 |
| **Mean across classes** | — | — | — | — | 0.534 | 0.559 | 0.382 | 0.396 | 0.375 | 0.424 | 0.326 | 0.352 |

Macro means weight each class equally. Correlations describe association, not causal improvement in AP.
