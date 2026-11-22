# PETR keyframe selection

This utility ranks nuScenes validation keyframes for internal PETR analysis.
It favors frames containing a high-confidence clean detection, a
high-confidence poorly localized detection, and GT objects visible in at
least two cameras, while penalizing false positives and missed GT.

Run it from the PETR repository root. To rank every keyframe in one scene:

```bash
python tools/analysis_tools/select_petr_keyframe.py \
  --scene-name scene-0103 \
  --score-threshold 0.4
```

To discover a scene and keyframe by scanning a deterministic subset:

```bash
python tools/analysis_tools/select_petr_keyframe.py \
  --max-samples 200 \
  --seed 0 \
  --score-threshold 0.35 \
  --classes car pedestrian truck
```

When `--classes` is supplied, clean/poor matches, false positives, missed GT,
multi-camera visibility, and ranking are restricted to those classes. Use the
same class list with `trace_query.py`.

Use `--score-threshold 0.5` for a stricter shortlist. The default definitions
are clean BEV center error <= 1 m and poor localization >= 2 m, with matches
limited to 4 m. These are configurable with `--clean-error`, `--poor-error`,
and `--max-match-distance`.

Outputs are `ranked_keyframes.csv` and `selection.json`. The latter contains
the selected scene name, zero-based `scene_frame_index`, one-based
`scene_frame_number`, dataset index, and stable nuScenes sample token for the
subsequent query tracing experiments.
