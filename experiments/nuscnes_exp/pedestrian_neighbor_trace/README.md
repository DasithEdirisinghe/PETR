# Crowded pedestrian group analysis

## PETR pedestrian image-alignment hypothesis audit

Run the validation-wide 4 m comparison on the HPC login or compute node:

```bash
bash experiments/nuscnes_exp/pedestrian_neighbor_trace/run_pedestrian_image_alignment.sh
```

The wrapper uses the `streampetr` conda environment by default and PETR's
saved validation `results_nusc.json`; override these with `CONDA_ENV`,
`RESULTS_JSON`, `DATA_ROOT`, or `OUTPUT_DIR`. The analysis includes every
officially filtered pedestrian GT. For each GT it records the official greedy
4 m match, the within-4 m prediction with maximum 3D IoU, and the within-4 m
prediction with maximum projected 2D IoU. Projection uses one camera chosen
from the GT alone by its largest visible projected area.

Outputs under `outputs/pedestrian_image_alignment_4m/` are `all_gt.csv`,
`geometrically_better_cases.csv`, `summary.json`, `summary.md`,
`visibility_distribution.csv`, and paired 3D/2D IoU delta plots. A geometric
case requires an alternative to have higher 3D IoU, higher projected 2D IoU,
and smaller absolute center error along the camera-to-GT ray than the official
TP. Confidence is recorded but is not a condition. This is a GT-informed
geometric diagnostic. Projected box overlap does not measure which image
features a PETR query used, and the CSV marks alternatives that were already
official TPs for another GT.

## Early-FP pedestrian audit (4 m and 2 m)

Run `python experiments/nuscnes_exp/pedestrian_neighbor_trace/early_fp_audit.py`
from the PETR environment. Use `--threshold 2` for the comparison run. The
script reuses official nuScenes filtering and the repository's official greedy
matching replay. It writes `outputs/early_fp_audit/4p0/` (or `2p0/`):

- `fps.csv`: **all** baseline pedestrian FPs, their score, global rank, box,
  nearest GT, and whether they precede the last TP;
- `fp_gt_relations.csv`: every pedestrian GT in each early FP's sample, its XY
  distance, and its baseline claimant, if any;
- `review_queue.csv`: a fixed rank-stratified image-review sample for each
  *geometric candidate* type; `review.html`: six-camera box projections for
  those cases (red FP, blue GT, yellow claimant);
- `review_labels.csv`: one row for **every early FP**, left blank initially;
- `category_counts.csv`: counts and percentages of the five reviewed diagnoses.
  With partial review, percentages use **reviewed FPs only**; full early-FP
  percentages appear only when every early FP has a label;
- `summary.json`: full-set TP/FP counts, early-FP candidate counts, baseline AP,
  and explicitly separate reviewed counts.

The geometric candidate types are `claimed_gt_within_threshold`,
`no_gt_within_threshold`, and `other_unmatched`. They are **not physical error
categories**. Review camera evidence and enter exactly one final label:
`genuine_duplicate` (FP and claimant represent the same pedestrian),
`association_ambiguity` (FP plausibly represents a different nearby person),
`likely_localization_error` (plausible same GT person, center outside the
threshold), `background_spurious` (no plausible pedestrian), or `uncertain`.
Record the corresponding GT ID and evidence notes where possible. A close GT
alone never proves a genuine duplicate, and an outside-threshold center alone
never proves localization error. Use `uncertain` if visibility is inadequate.
`fps.csv` also records how many baseline TPs each FP precedes in global score
order (`later_tp_count`), so category frequency can be interpreted alongside
where the FPs affect AP.

Rerun the command after entering labels to refresh counts. Only when **every
early FP** has a final label does the script compute a category-specific
GT-informed oracle AP. Each oracle moves that category's baseline FPs to the
end of the global score order, retains all boxes, and recomputes matching and
official AP. The gains are diagnostic and cannot be added across categories.
The fixed review queue is for an initial audit; its label counts are not
presented as validation-wide prevalence. For further cases, increase
`--review-per-kind` or use the saved prediction IDs to inspect them in the
camera gallery. The label file is preserved on rerun.

This uses saved PETR validation predictions; no GPU inference is needed. Run
in the PETR environment with `nuscenes-devkit`, `scipy`, and its dependencies:

```bash
python experiments/nuscnes_exp/pedestrian_neighbor_trace/analyze.py
```

The default paths use baseline PETR `results_nusc.json` and nuScenes trainval
metadata under `../RoboBEV/data/sets/nuscenes`. Override them with
`--results-json` and `--data-root`; use `--sample-token TOKEN` for a pilot.
The script uses the evaluator's filtered GT and retained submitted predictions.

For each validation keyframe, pedestrian GTs are connected when their BEV
centers are at most `--cluster-distance` apart (default 4 m). If several
crowded groups exist, the script selects the largest, breaking ties by mean
pairwise GT distance. Thus there is **at most one group per keyframe**, and a
group has at least two pedestrian GTs. Group membership is fixed across all
thresholds.

Every retained pedestrian prediction within `--prediction-radius` (default
6 m) of any group GT is included, as are predictions officially matched to a
group GT even if farther away. This includes TPs and FPs. Official matching is
replayed across the **entire validation set**, separately at the configured
0.5, 1, 2, and 4 m thresholds. The group is extracted only after global
matching, so an outside prediction can affect the official assignment.

Outputs in `outputs/`:

- `groups.jsonl`: one JSON object per selected keyframe, with all group GTs,
  local predictions, their confidence scores and global class ranks, the full
  prediction–GT center-distance/BEV-IoU/3D-IoU matrix, and official and
  counterfactual assignments at each threshold.
- `groups.csv`: one row per group and threshold, with official TP/FN counts,
  local prediction outcomes, diagnostic matching counts, and a pattern label.
- `group_summary.json`: selected keyframe count and pattern counts per threshold.
- `gt_diagnostics.csv`: one row per group GT and threshold. It reports the
  official match, nearest and highest-overlap predictions, their scores and
  official destinations, counts of predictions within 1/2/4 m, and whether
  a local reassignment covers a missed GT.
- `prediction_diagnostics.csv`: one row per local prediction and threshold,
  with score/rank, TP/FP status, nearest and highest-overlap group GT, and
  whether either differs from its official assignment.
- `interpretation_summary.json`: GT and prediction category counts by
  threshold. This is the main spatial interpretation summary.

For an existing `groups.jsonl`, regenerate only the interpretation without
loading nuScenes or running matching again:

```bash
python experiments/nuscnes_exp/pedestrian_neighbor_trace/interpret.py
```

The counterfactuals use one-to-one matching within the selected group and
maximize the number of pairs first. One minimizes center distance subject to
the official distance threshold; the other maximizes BEV IoU subject to the
same distance threshold and `--min-bev-iou` (default 0.1). These are local
diagnostics, **not official nuScenes AP**. `assignment_sensitive` means local
distance matching covers more GTs than the official global score-ordered match;
it does not by itself prove a model-level identity error. A prediction matched
to an outside GT can appear in the local counterfactual.

GT interpretation separates a missed GT with **no prediction within the
threshold** from one with a **close prediction claimed elsewhere**. Matched
GTs are separated by whether the assigned box has BEV IoU at least 0.1, and
whether a different prediction overlaps substantially better. Predictions
are separately labelled FP, TP in the group, or TP outside it. These are
observable geometric/assignment patterns, not a causal explanation of PETR's
internal behavior. For example, a box within 4 m may have zero pedestrian
overlap, and a local counterfactual may reuse a box claimed by another GT.

The previous selected-prediction `cases.csv`, `cases.json`, and `summary.json`
files, if present, are from the older experiment and are not updated by this
version. Use the `groups.*` files for the group analysis.

## Pedestrian localization capacity and confidence order

Run the validation-wide, pedestrian-only diagnostic in the `petr` environment:

```bash
python experiments/nuscnes_exp/pedestrian_neighbor_trace/localization_capacity.py
```

This reuses the saved baseline predictions and nuScenes evaluator filtering;
PETR inference and a GPU are not needed. At 0.5, 1, 2, and 4 m, it adds
predictions in PETR confidence order and computes the maximum one-to-one
number of pedestrian GTs coverable by each prefix. It also replays nuScenes'
confidence-ordered nearest-GT matching and shows an ideal GT-aware prediction
order as an upper bound. The outputs in `outputs/localization_capacity/` are:

- `pedestrian_localization_capacity.csv`: full-set localization capacity,
  top-budget coverage, ranking efficiency, and greedy-assignment efficiency;
- `pedestrian_top_k_coverage.csv`: sampled confidence-prefix curves;
- `pedestrian_top_k_coverage.svg` (and `.png` when `rsvg-convert` is installed):
  the three coverage curves for each distance threshold;
- `source.json`: prediction source, filtering, and metric definitions.

Localization capacity is the GT fraction coverable by any retained prediction
under a strict center-distance threshold. Ranking efficiency is the area under
the maximum-matching coverage curve divided by the ideal-order area for the
top `min(number of predictions, number of GTs)` predictions. Greedy-assignment
efficiency compares the official score-order nearest-GT curve with the
maximum-matching curve at the same confidence prefixes. These are diagnostic
coverage measures, not official nuScenes AP or BEV-IoU measurements.

## Review keyframes in BEV and camera images

After generating the group and interpretation files, run:

```bash
python experiments/nuscnes_exp/pedestrian_neighbor_trace/visualize.py
python experiments/nuscnes_exp/pedestrian_neighbor_trace/plot_research_summary.py
```

Open `outputs/visualizations/index.html`. By default it shows 30 crowded and
30 non-crowded pedestrian keyframes, plus 10 crowded and 10 non-crowded
keyframes each for car and bicycle (100 total). The global Class and Crowding
controls switch among them; matching mode and the 0.5, 1, 2, and 4 m threshold
controls apply to every class. Crowding is based on another same-class GT
within 2 m for pedestrian/bicycle or 4 m for car, independently of the
evaluation threshold. The crowded pedestrian cases are selected across the
five diagnostic categories using the 2 m
nearby-count radius. Each view includes **all retained pedestrian
predictions within the group's 6 m collection radius**, including FPs, with
score and local rank. Blue solid boxes are GTs; purple dashed boxes are
predictions, with darker/thicker outlines for higher confidence rank. Use the
controls at the top of the HTML page to highlight official TPs at 0.5, 1, 2,
or 4 m. Green rings and links in BEV, and green prediction wireframes and
assignment labels in the camera overlay, follow the selected threshold.
Each frame's table compares TP/FP outcomes and assigned GTs at all four
thresholds side by side, along with score rank and nearest group GT. The TP
controls are independent of `--threshold`, which sets the nearby-count column.
The figure above the cases compares official full-validation pedestrian AP
from the same saved prediction run with GT match fractions in the displayed
groups and all analyzed crowded groups. `pedestrian_ap_scene_summary.csv`
contains the figure values, and `non_nearest_tp_review.csv` lists official TPs
assigned to a GT other than their nearest displayed group GT. The displayed
groups are selected visible, close-range examples, so their match fraction is
not official AP. Geometric non-nearest matches need image-level review before
making an instance-identity claim.
Each case shows one camera image followed by its camera-aligned BEV map. The
GT and prediction overlay buttons in the global toolbar apply to every camera
image. Click the BEV
map to open it at full size and inspect ego position and confidence ranks.

Every GT in a selected group must have nuScenes visibility token 4 (80–100%)
in the visibility CSV and be within 20 m horizontal distance of the keyframe
ego position. Adjust that limit with `--max-ego-distance`. The index also shows the camera view containing the
most projected group GT centers. BEV is rotated so that camera forward points
up and camera right points right, including in the ego context inset. Each
selected GT center must also project inside that camera image. The camera
overlay labels official TP predictions with their assigned GT. Use the BEV
view for all GT IDs and prediction confidence ranks. Visibility token 4 describes the annotation,
so check the camera crop when deciding whether a particular view is clear.
In the camera overlays, solid blue dots show GT centers and hollow blue dots
show the center of each GT box's bottom face. The near corners can project
below the visible feet because the box has nonzero depth.

The projected 3D wireframes can extend beyond a person's image silhouette.
They are global nuScenes 3D boxes transformed to the camera pose, not 2D
segmentation masks. The BEV/3D IoU code uses nuScenes width/length convention
(length along local x, width along local y). Rerun `analyze.py` and `interpret.py`
after changing this geometry to refresh existing IoU values and diagnostic
categories in `outputs/groups.jsonl` and the CSV summaries.

The black marker shows the ego vehicle's actual keyframe LIDAR_TOP position.
When it is outside the local zoom, a boundary marker points toward it. The
inset shows ego and the GT-group center at a common metric scale, with a
short line for ego heading. Ego poses come from nuScenes metadata; the selected
positions are cached in `ego_positions.json` after the first render.

Use `--threshold 0.5`, `1`, `2`, or `4` and a separate `--output-dir` to inspect
the same selection process at another threshold. The `manifest.csv` lists
sample tokens and the pattern used to select each frame.

## SAM 2 person correspondence pilot

The segmentation pilot starts from the same ten selected camera frames:

```bash
python experiments/nuscnes_exp/pedestrian_neighbor_trace/sam_pilot.py --prepare-only
```

Open `outputs/sam_pilot/prepare.html`, click each visible pedestrian's torso,
and add its sample token, an independent person ID (for example `A`), and the
pixel coordinates to `outputs/sam_pilot/manual_people.csv`. One row per person
is sufficient. These manual points define image people independently of the
nuScenes GT and PETR boxes. The GT-derived point and box prompts in
`prompt_suggestions.csv` are exploratory controls.

Install [SAM 2](https://github.com/facebookresearch/sam2) in the GPU environment
and provide a matching config and local checkpoint. For example:

```bash
python experiments/nuscnes_exp/pedestrian_neighbor_trace/sam_pilot.py \
  --sam-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --sam-checkpoint /path/to/sam2.1_hiera_tiny.pt
```

Review `outputs/sam_pilot/index.html` and `person_results.csv` for whether
each independently clicked person was segmented. `correspondence.csv` has one
row per SAM person mask and projected GT or prediction box. It includes 2D
center-in-mask and box/mask overlap, prediction score and rank, and official
GT assignment at each nuScenes distance threshold. Compare a prediction's
image-mask correspondence with its official assigned GT. SAM's predicted IoU
is its own quality estimate, not measured segmentation accuracy. A human
review is needed to confirm separate people in crowded images. The 2D mask
matrix is diagnostic and does not replace the official 3D evaluator.
# Pedestrian confidence re-ranking oracle

Run `python experiments/nuscnes_exp/pedestrian_neighbor_trace/reranking_oracle.py`
from the PETR repository root in an environment with `nuscenes-devkit` installed.
Use `--results-json`, `--data-root`, and `--output-dir` to override the defaults.

The script uses the official filtered validation prediction set and pedestrian
GTs. At each center-distance threshold (0.5, 1, 2, 4 m), it reports:

- Maximum-cardinality one-to-one GT coverage using all retained predictions.
- Full-set TPs from confidence-ordered official greedy matching.
- Full-set TPs and official nuScenes pedestrian AP after GT-informed re-ranking.

The re-ranking assigns unique high scores to predictions selected by maximum
matching, then gives all other predictions lower scores. Box coordinates and
the prediction set stay fixed. Because official greedy matching can associate
a selected prediction with a different GT, its realized TP count can remain
below maximum-cardinality coverage. The script reports this remaining gap.
AP is calculated with nuScenes `accumulate` and `calc_ap`, but the oracle scores
use GT information and are diagnostic only. Results are written to
`outputs/reranking_oracle/pedestrian_oracle_comparison.csv`.

The comparison CSV also reports the mean and median **original PETR score** of
matched TPs, their mean XY center error, and their mean oriented BEV IoU. These
are calculated separately for baseline and re-ranked official greedy matches.
The `shared_gt` columns repeat the geometry comparison on GTs matched in both
runs; this avoids comparing different GT populations when assessing box quality.
The artificial re-ranking scores are not used for the score summary.

To add the same re-ranking comparison to the 60-frame gallery, run
`reranking_oracle.py` again after `visualizations/manifest.csv` exists. It writes
`outputs/reranking_oracle/gallery_oracle_assignments.json` using full-frame
matching. Then run `visualize.py` to regenerate `visualizations/index.html`.
The gallery has global controls for baseline versus GT-informed matching and
for 0.5, 1, 2, and 4 m. It shows the same GT and predicted boxes in both modes;
the green TP links, assignment table, and displayed prediction ranks change.
The oracle rank is recalculated separately for each threshold. The table's
standalone oracle-rank column uses 4 m; each threshold cell includes its own
rank. Local displayed predictions remain the original 6 m neighborhood.

Randomly sampled cases use official nuScenes filtered boxes. A non-crowded
case has one target GT with no same-class GT within its crowding threshold.
A crowded case has a target pair within the threshold. The gallery also draws
every other same-class GT with visibility token 4 that is within 20 m of ego
and projects into the chosen camera. All displayed GT boxes are blue in the
camera and BEV views. The nearby prediction set stays fixed across matching
thresholds and methods. Full-frame TPs are green, including predictions
matched to a GT outside the displayed group; FPs are purple. BEV labels show
each prediction's rank among all retained same-class predictions in that frame.
GT labels show only their G number.
Crowding describes the target, while the assignment tables include all these
visible GTs. The display includes
all retained same-class predictions within 8 m XY of those GTs. Baseline and
oracle TP assignments are replayed on **all same-class predictions and GTs in
that frame** before the display radius is applied. A displayed prediction may
therefore be labeled as matched to a GT outside the visible group. Random
selection uses seed 42; adjust the pedestrian sample size with `--count` and
`--crowded-count`, the car/bicycle size per crowding group with `--extra-count`,
the seed with `--extra-seed`, or the displayed prediction radius with
`--extra-prediction-radius` (minimum 4 m). `--extra-count 0` generates only the
pedestrian gallery. All random groups require the nuScenes devkit and SciPy
in the PETR environment.

To select frames where the **final greedy** GT assignment differs between
baseline PETR and the 3D IoU rerank, run:

```bash
python experiments/nuscnes_exp/pedestrian_neighbor_trace/visualize.py --selection-mode changed --selection-threshold 2
```

This uses `assignment_transition/gt_transitions.csv`. Each card identifies
the changed target GT and lists its baseline and 3D IoU rerank prediction
IDs, original PETR scores, 3D IoU, and XY error. Use
`--change-filter improved` for higher-IoU replacements or
`--change-filter b1` for higher-IoU replacements with lower original
confidence. The BEV colors each prediction matched to a displayed GT green
under the currently selected mode and threshold. Use `--output-dir` to keep
the changed-case gallery separate from the random gallery.

To select frames with many high-ranked baseline FPs before the last TP, while
keeping the same camera, BEV, class, crowding, matching-method, and threshold
controls, run:

```bash
python experiments/nuscnes_exp/pedestrian_neighbor_trace/visualize.py \
  --selection-mode early_fp --selection-threshold 1
```

The HTML shows five cases per page so changing the matching mode or distance
threshold updates only the visible camera and BEV SVGs. Use Previous cases and
Next cases to inspect the rest.

To add StreamPETR predictions to those **same selected frames** with a model
toggle, run this after generating the PETR gallery:

```bash
python experiments/nuscnes_exp/pedestrian_neighbor_trace/add_models_to_gallery.py
```

The default input is the saved StreamPETR R50 428-query validation
`results_nusc.json` under `../StreamPETR/test/`. Each model is filtered and
matched independently on the full frame; displayed GTs, camera images, and
frame selection stay fixed. The script writes model-specific camera and BEV
SVGs beside the PETR assets and updates `index.html` with a model toggle and
model-specific assignment tables. It saves the original PETR-only HTML as
`index_petr_only.html` and model paths in `comparison_models.json`.

For future models, add a saved nuScenes result JSON by name:

```bash
python experiments/nuscnes_exp/pedestrian_neighbor_trace/add_models_to_gallery.py \
  --model StreamPETR-ViT=/path/to/stream_vit/results_nusc.json \
  --model PETR-ViT=/path/to/petr_vit/results_nusc.json
```

Previously added models are retained. The model name becomes a toggle and a
filename suffix. Raw confidence values are interpreted within each model;
cross-model comparisons should use ranks, matches, and geometry.

For each class and frame, selection replays baseline matching using every
officially filtered same-class prediction and GT. It requires at least two
full-frame TPs, more full-frame FPs before the last TP than TPs, and at least
two such early FPs near a visible target GT (8 m by default), where they
outnumber nearby TPs. At least one early FP center must project into the
chosen camera. The highest-count usable cases are selected separately
for crowded and non-crowded pedestrian, car, and bicycle targets. The card and
`manifest.csv` show the PETR selection
counts. Change `--min-early-fp` or `--extra-prediction-radius` to adjust local
case selection. Other thresholds remain available for reviewing the same
frames; they do not change which frames were selected.

## All-class re-ranking AP

To compute the same confidence-versus-best-3D-IoU Pearson and Spearman
correlations for the saved StreamPETR predictions and compare them class by
class with PETR, run:

```bash
python experiments/nuscnes_exp/pedestrian_neighbor_trace/compare_model_correlations.py
```

This reuses `confidence_quality_correlation.py` with StreamPETR's saved
`results_nusc.json` and the same nuScenes filtering. It writes
`outputs/confidence_quality_correlation/model_comparison/` with a StreamPETR
`class_correlation.csv`, a side-by-side comparison CSV, and a Markdown table.
The calculation does not require a GPU. The all-prediction and positive-IoU
subsets are separate; no TP matching or distance threshold enters this table.

To compare baseline and 3D IoU re-ranking precision–recall curves, run:

```bash
python experiments/nuscnes_exp/pedestrian_neighbor_trace/plot_pr_curves.py
```

This writes `outputs/all_class_reranking_oracle/pr_curves/index.html`, one
four-threshold SVG per class, `pr_summary.csv`, and sampled PR coordinates in
`pr_curves.json`. Each threshold panel shows both curves and the maximum
one-to-one localization recall. The CSV records achieved maximum recall,
peak interpolated precision at positive recall, full-prediction-set precision,
and AP for both methods. Peak precision can occur at very low recall, so the
full curve and AP are more informative. The re-ranking uses GT information
separately at each threshold and is diagnostic.

Run `python experiments/nuscnes_exp/pedestrian_neighbor_trace/all_class_reranking_oracle.py`
from the PETR root in the environment with `nuscenes-devkit`. This repeats the
fixed-box experiment for all ten nuScenes detection classes. It loads the
official filtered validation boxes once, builds a separate GT-informed score
order for each class and distance threshold, and computes nuScenes AP with
`accumulate` and `calc_ap`. It reports maximum one-to-one GT coverage and
full-set baseline/oracle greedy TP counts as context.

Outputs are `class_threshold_comparison.csv` (40 class/threshold rows),
`class_mean_ap_comparison.csv`, `threshold_mean_ap_comparison.csv`, and
`summary.json` under `outputs/all_class_reranking_oracle/`. The baseline AP for
each class and threshold is checked against the saved source
`metrics_summary.json`. The oracle mean AP averages 40 independent
GT-informed orderings; it is a diagnostic figure, not the AP of one deployable
scoring model or one nuScenes results file. Use `--classes car pedestrian` to
run a subset for a quick check.

Both the pedestrian and all-class scripts also report a **minimum-distance
localization oracle**. At each threshold, it maximizes one-to-one GT matches
first, then minimizes total ground-plane center distance among matchings of
that size. The CSV contains its theoretical total and mean center error and
mean BEV IoU. `min_distance_rank_*` columns then show what happens when those
selected predictions are given GT-informed priority and official greedy
matching/AP are rerun. The greedy assignment can differ from the theoretical
minimum-distance pairing. The earlier `oracle_rank_*` columns remain as the
cardinality-only selection for comparison.

The same scripts now include a **3D IoU localization oracle**. For each
center-distance threshold, it first maximizes the number of one-to-one GT
matches, then maximizes total oriented 3D box IoU among those matchings.
Eligible pairs still use the nuScenes XY center-distance threshold; zero-IoU
pairs can therefore remain eligible. `max_iou3d_oracle_*` columns describe
the theoretical selected pairs. `max_iou3d_rank_*` columns describe actual
greedy TP assignments and AP after the selected predictions are prioritized.
These pairings can differ, and the AP is a GT-informed diagnostic. The 60-frame
gallery adds a **3D IoU re-rank** switch after rerunning `reranking_oracle.py`
and `visualize.py` in the PETR environment.

Running `all_class_reranking_oracle.py` also writes a single
`outputs/all_class_reranking_oracle/tp_geometry_by_class.svg` with each class
on its own row. It compares
the mean 3D IoU and mean XY center error of **realized greedy TPs** for the
PETR baseline, cardinality re-rank, distance re-rank, and 3D IoU re-rank at
0.5, 1, 2, and 4 m. The plotted values and TP counts are in
`tp_geometry_plot_data.csv`. Because the methods can match different GT sets,
these means describe each method's own TP assignments.

For each score ordering, the comparison CSV reports the mean **original PETR
score**, XY center error, BEV IoU, and 3D IoU of the predictions actually matched as
TPs by greedy evaluation. It also reports those measures for the three
theoretical one-to-one assignments. Theoretical assignments have a match count
and geometry, but no AP until their selected predictions are re-ranked and
passed through official matching. Re-run the scripts after updating them to
populate these added columns.

## Confidence versus geometric quality (RARE-style diagnostic)

Run `python experiments/nuscnes_exp/pedestrian_neighbor_trace/confidence_quality_correlation.py`
from the PETR repository in the nuScenes environment. It uses the same
officially retained PETR predictions as the oracle analyses. For each
prediction, it records the original PETR score and the maximum oriented 3D
IoU with any same-class GT in that frame. A prediction with no overlapping
same-class GT receives quality zero. This is a per-prediction quality target,
not a one-to-one TP assignment: several predictions may have the same best GT.

`outputs/confidence_quality_correlation/prediction_quality.csv` contains the
auditable per-prediction values. `class_correlation.csv` reports Pearson and
Spearman correlations per class, both over all retained predictions and over
the positive-IoU subset. The all-prediction result reflects score separation
from zero-quality boxes; the positive subset focuses on ordering among boxes
that overlap a GT. Correlations are undefined if a subset has fewer than two
predictions or constant scores/qualities, and appear as empty CSV cells. Use
`--classes pedestrian` for a pedestrian-only run.

The same command also writes `center_distance_correlation.csv`. It compares
original PETR confidence with XY center error for three subsets: all
predictions in frames with a same-class GT (nearest-GT distance), predictions
whose nearest GT is within each 0.5/1/2/4 m threshold (nearest-GT distance),
and baseline confidence-greedy TPs at each threshold (their **assigned** GT
distance). Predictions in frames without any same-class GT have no defined
center distance and are excluded from this particular correlation; their
count is reported. The CSV reports score-versus-**negative distance**
correlations only. Positive values indicate that higher confidence
accompanies smaller center error.
`prediction_quality.csv`
includes nearest GT and threshold-specific TP assignments for audit.

# Class-wise assignment transition analysis

Run from the PETR repository root after the all-class oracle experiment has produced
`outputs/all_class_reranking_oracle/class_threshold_comparison.csv`:

```bash
python experiments/nuscnes_exp/pedestrian_neighbor_trace/assignment_transition.py
```

The script compares final greedy baseline and 3D-IoU-oracle assignments using
the same filtered predictions. It validates TP totals and TP geometry against
the saved all-class experiment before writing
`outputs/assignment_transition/{primary_2m_summary.csv,all_threshold_summary.csv,gt_transitions.csv}`
and three SVG diagnostic plots. B1 means a changed assignment with higher 3D
IoU and lower *original PETR* confidence. Rates for B1 use GTs matched by
both methods as denominator. The fixed-subset B1 rate uses GTs matched by both
methods at all four thresholds. Oracle scores are never used as original scores.

To compare B1 frequency in crowded and non-crowded GTs, run:

```bash
python experiments/nuscnes_exp/pedestrian_neighbor_trace/crowding_b1.py
```

This reads the saved GT-level transition CSV and writes
`outputs/assignment_transition/crowding_b1_summary.csv` (40 class-threshold
rows). A GT is crowded when at least one other *same-class filtered GT* in its
frame has XY center distance strictly below the fixed crowding threshold:
2 m for pedestrian, bicycle, motorcycle and traffic cone; 4 m for all other
classes. The crowding label stays fixed across evaluation thresholds. B1
percentages use GTs matched by both methods within each crowding group;
undefined rates are written as `NaN`.
