# Fixed 4 m true-positive retention

From the PETR repository root, run:

```bash
python3 experiments/nuscnes_exp/fixed_4m_tp_retention/plot_tp_retention.py
```

This uses the saved classwise range summary from
`range_conditioned_localization`, whose error-band percentages retain enough
precision to recover exact counts. No inference, dataset load, or nuScenes
devkit is required. Outputs are in `outputs/baseline/`:

- `fixed_4m_tp_retention.svg`: one heatmap panel per class plus all classes.
- `fixed_4m_tp_error_bands.svg`: disjoint 0–0.5, 0.5–1, 1–2, and 2–4 m
  error bands; percentages sum to 100% within each class/range row.
- `fixed_4m_tp_retention.md`: exact counts and percentages.
- `fixed_4m_tp_retention.csv`: machine-readable cells.
- `fixed_4m_tp_error_bands.csv`: exact disjoint-band counts.

The denominator in each class/range row is its **original 4 m true-positive
count**. For example, the 1 m cell counts how many *of those same matched
predictions* have center error <1 m. This fixed-pair analysis differs from
official independent matching at each threshold, and it is not AP.

The same heatmap can be generated from the saved StreamPETR and OPEN per-match
files, using only their `match_threshold_m == 4.0` rows:

```bash
python3 experiments/nuscnes_exp/fixed_4m_tp_retention/plot_other_models.py --model StreamPETR
python3 experiments/nuscnes_exp/fixed_4m_tp_retention/plot_other_models.py --model OPEN
```

Their disjoint and cumulative SVGs, CSVs, and source notes are saved under
`outputs/streampetr/` and `outputs/open/`. The same class/range bins and color
scale are used for PETR, StreamPETR, and OPEN.

## PETR visibility and nearby-GT companion heatmaps

From the PETR repository root, in the PETR environment with the nuScenes
devkit, NumPy, matplotlib, and `pyquaternion` installed:

```bash
python3 experiments/nuscnes_exp/fixed_4m_tp_retention/plot_tp_context_heatmaps.py
```

This separate script uses the saved baseline PETR `results_nusc.json` and
nuScenes validation GT; it does not run inference. It replays the same
evaluator-filtered, confidence-ordered `<4 m` TP matches, then writes:

- `outputs/baseline/tp_mean_visibility_heatmap.png` and `.svg` — overview
  of four-band GT visibility composition, replacing the former mean-visibility
  heatmap at the same paths. Each cell shows a 100% stacked bar, TP total,
  and the count and within-cell percentage for visibility tokens 1–4.
  Dark red means 0–40% visible; the colors lighten toward 80–100% visible.
- `outputs/baseline/tp_visibility_bands_<class>.png` and `.svg` — the same
  chart as a larger, easier-to-read figure for each class and all classes.
- `outputs/baseline/tp_other_gt_closer_heatmap.png` and `.svg` — percentage
  of TPs whose prediction center is closer in BEV to a different
  evaluator-eligible same-class GT center than to its assigned GT center.
  Each TP contributes 100% if true, 0% otherwise, including when there is
  no other GT in that sample.
- `outputs/baseline/tp_context_by_class_range_error.csv` — counts, all four
  visibility categories, both plotted metrics, mean and median nearest-other
  distance, and nearby or overlapping-other-GT rates.
- `outputs/baseline/tp_context.csv` — one row per original 4 m TP with the
  underlying values and conflict flags.

To redraw only the visibility figures from the existing summary CSV, without
loading nuScenes or rerunning matching, run:

```bash
python3 experiments/nuscnes_exp/fixed_4m_tp_retention/plot_tp_visibility_bands.py
```

The visibility renderer needs `rsvg-convert` or Python `cairosvg` to write PNG;
it always writes SVG. Each visibility percentage uses the full TP count in its
cell as the denominator. The closer-other-GT heatmap also uses the full TP
count in each cell. A strict
distance comparison is used: ties score 0% in the neighboring-GT heatmap.
The nearest-other distance retained in the CSV is defined only when another
same-class GT exists. Visibility is an annotation-level interval, so its
midpoint retained in the CSV is an estimate rather than an exact visible
percentage or a camera-level occlusion measurement. These descriptive plots do not establish
why PETR made a particular localization error.
