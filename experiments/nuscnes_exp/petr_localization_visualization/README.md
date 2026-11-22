# PETR localization videos

This pipeline uses the **saved PETR validation predictions** at
`../localization_oracle/outputs/baseline/baseline/results_nusc.json`. No PETR
inference is needed. The prior image geometry run used this file and recorded
116,212 matched 4 m TPs across the full validation set (see
`../image_geometry_localization/outputs/petr/run.json`). The saved inference
file is about 593 MB; the nuScenes data root is `../../../../RoboBEV/data/sets/nuscenes`.

Run from this directory in the PETR environment with `nuscenes-devkit`,
`numpy`, `pyquaternion`, OpenCV, and `ffmpeg` with `libx264` installed. The default run selects 10
validation scenes, all 10 detection classes, all keyframes and six cameras:

```bash
python pipeline.py
```

Run the pilot first:

```bash
python pipeline.py --scenes scene-0103 --classes car \
  --output-dir outputs/pilot
```

`scene-0103` is an example; use any validation scene name. To choose the
10 scenes automatically but inspect the selected scene metadata before
encoding videos, use `--metadata-only`. You can then pass those scene names
with `--scenes` to the full run.

Important options: `--scenes`, `--scene-count`, `--classes`, `--cameras`,
`--cell-width`, `--cell-height`, `--bev-width`, `--fps`, `--results-json`,
`--data-root`, `--output-dir`, and `--metadata-only`.

The legend sits directly under the BEV panel at the right of each frame.
The BEV panel and legend together match the camera mosaic height.

The output directory contains one H.264 MP4 and a `_first_frame.png` preview
per scene and class, `legend.png`, and
`metadata/selected_scenes.json`, `metadata/matched_4m_tps.csv`, and
`metadata/video_manifest.csv`. Scene and frame order follow the nuScenes
sample chain. Empty-TP frames are retained in videos.

Matching uses the nuScenes evaluator's filtered GT and predictions, class
names, center-distance function, confidence ordering, nearest available GT,
and strict `< 4 m` threshold. The displayed GT and prediction boxes come
only from those pairs. Error bins are `[0, 0.5)`, `[0.5, 1)`, `[1, 2)`, and
`[2, 4)` m. Distance bins use the corresponding half-open intervals. Numeric
IDs are stable within each scene by GT instance token. Camera boxes use the
camera sample-data ego pose and calibration; BEV uses the keyframe LIDAR_TOP
ego pose and a fixed 100 m by 100 m metric view.

Check the pilot's `car_first_frame.png` and `car.mp4` before starting the full
set. The PNG checks rendering independently of MP4 decoding. The MP4 is
encoded with `ffmpeg`/`libx264` and `yuv420p` for broad viewer compatibility.
