# PETR-R50 query rules and an added score loss for all classes

Status: opt-in first implementation in `projects/configs/petr/petr_r50dcn_gridmask_p4_ray_rule.py`. It inherits the trained two-GPU, global-batch-eight PETR-R50 configuration and loads its epoch-24 checkpoint for a short fine-tuning run. The rule thresholds, loss weight, and effects on training have not been validated. The ordinary PETR config, Hungarian cost, classification loss, and regression loss remain unchanged.

The first configured run is **one epoch**. `RayRuleDiagnosticsHook` is enabled only in this config and writes `ray_rule_diagnostics_epoch_1.json` in the run directory. The report contains mean per-iteration sums of all six decoder-layer classification, box, and weighted rule losses; sampled gradient norms of these three terms at the shared decoder output; preferred and inferior case counts and BCE means; Hungarian-conflict case means; radial- versus tangential-dominant inferior means; ranking pair counts and penalties; and per-class preferred/inferior counts and BCE means. The gradient norm sample runs every 50 training iterations to limit overhead. These measurements help choose a rule-loss weight; they do not establish improved AP. Role BCE means are shown before and after the outer rule-loss weight, and the grouped rule loss is also normalized by active class and case group, so individual case means do not sum directly to the final scalar loss.

An independent freeze experiment uses `petr_r50dcn_gridmask_p4_ray_rule_frozen_backbone.py` and its own output directory. The optional `model.parameter_freeze_cfg` takes exact module names in `freeze_modules` and optional descendant names in `unfreeze_modules`. A frozen module's parameters stop receiving updates and it stays in eval mode; unfreezing restores each parameter's original `requires_grad` state. The provided variant freezes only `img_backbone`, keeping `img_neck`, position encoding, query embeddings, decoder, and heads trainable. Both variants start from the original PETR-R50 epoch-24 checkpoint. The running all-trainable experiment and ordinary PETR configs have no freeze policy.

The stricter `petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only.py` variant freezes `img_backbone`, `img_neck`, the image-to-decoder `input_proj`, the learned 3DPE `position_encoder`, and its `adapt_pos3d` feature adapter. Query reference points and embeddings, decoder layers, classification branches, and regression branches remain trainable. This is the variant for a decoder-and-head-only comparison; it has its own PBS script and output directory and starts from the same original checkpoint. PETR's sinusoidal positional encoding has no trainable weights in this configuration.

The decoder-only comparisons are named as a version series. Each config header states its base and change from the previous version:

| Version | Rule weight | Freeze policy | Checkpoint and schedule | Output |
|---|---:|---|---|---|
| `ray_rule_decoder_heads_only_v1.py` | 0.05 | Decoder/head-only | Original epoch 24; 1 epoch at 2e-5 | Historical decoder-only work directory |
| `ray_rule_decoder_heads_only_v2.py` | 0.075 | Same as V1 | Same as V1 | Separate V2 directory |
| `ray_rule_decoder_heads_only_v3.py` | Disabled | Same as V2 | Same as V2 | Separate V3 control directory |

V2 versus V3 isolates adding the rule loss at 0.075. V1 versus V2 isolates the outer rule-loss weight. All PBS scripts use seed 0 and global batch eight. The unversioned config names remain available for earlier submitted jobs.

### After the one-epoch comparison

The original epoch-24 checkpoint scored mAP 0.3277 and NDS 0.3782 on nuScenes validation. Decoder-only V2 (rule weight 0.075) scored mAP 0.3237 and NDS 0.3731; decoder-only V3 (no rule) scored mAP 0.3223 and NDS 0.3753. Thus the rule gave +0.0014 mAP but -0.0022 NDS against its matched control after one epoch. Every class improved at the 1 m AP threshold and declined at the 4 m threshold. This is a real trade-off in the measured validation result, not proof of the mechanism.

V2's rule term averaged 0.9663 per iteration, versus 2.0409 classification and 3.3134 box loss. Sampled decoder-output gradient norms were 0.0283, 0.0506, and 0.0770 respectively. About 47.2% of preferred entries disagreed with Hungarian matching. Within-epoch ranking inversions fell from approximately 16.2% in the first logging window to 14.2% in the last; Hungarian disagreement stayed high. These measurements show nontrivial rule supervision but do not establish that its gradients win on conflicting logits. The longer-run diagnostics now record that conflict directly.

Before interpreting a longer run, audit these two features of the current loss: (1) same-class preferred queries from *different GTs* can enter the pairwise margin, although both may be valid detections; (2) the only clear candidate for a GT receives its raw ray-quality target, which can be small and push its score down. Clear projected ownership currently uses overlap and an overlap margin; it does not require small depth error. These are plausible causes of the 4 m AP decline, not verified explanations. Avoid changing both rule design and outer weight in the same comparison.

V4 and V5 extend the matched decoder-only rule/control runs to **four epochs**, both starting afresh from the same original epoch-24 checkpoint. They use separate work directories, validate every epoch, and keep V2/V3's loss and freeze settings. V4 records rule case statistics and sampled decoder/logit gradient alignment each epoch. Compare their full AP curves, NDS, mATE, and within-epoch supervision trends before choosing a new weight or changing the rules.

## The decision we need to make

For every decoder-layer query prediction, decide whether it is a good candidate for a particular ground-truth (GT) object, a poorer candidate for that same object, a candidate for another object, or too uncertain to supervise. This is done for every GT class. Then compare that rule decision with PETR's current Hungarian match and the query's score for that GT class. The added loss should raise scores for credible, accurately localized representatives and lower scores for clear inferior candidates. A later training step may then make a different Hungarian assignment. We must measure that switch; it is a hypothesis, not a consequence guaranteed by the loss.

Official nuScenes TP/FP labels are evaluation outcomes produced from score order and center distance. They do not define query ownership for this loss. In particular, a 4 m TP can point toward a different GT, and an FP can be a better candidate for a GT already claimed by a higher-scoring prediction.

## PETR training behavior that the rules must work with

The R50-P4 configuration (`projects/configs/petr/petr_r50dcn_gridmask_p4.py`) uses 900 queries, ten sigmoid class outputs, and six decoder layers. For each layer, `PETRHead.forward` produces a class logit and encoded 3D box for every query. `PETRHead.loss` calls `loss_single` separately on each layer. There is no top-300 training selection: top 300 query-class scores are selected only in the final inference decoder.

For query q and GT g with class c_g, the current matcher minimizes a frame-wide one-to-one cost

\[
 C_{qg}=2F(\operatorname{sigmoid}(s_{q,c_g}))
       +0.25\lVert z_q^{1:8}-z_g^{1:8}\rVert_1.
\]

Here F is the configured focal matching cost. The first eight encoded box values contain center, log dimensions, and sine/cosine orientation. Velocity and IoU do not enter this match. The matching uses all GT classes jointly in the frame and is independently recomputed for each decoder layer (`core/bbox/assigners/hungarian_assigner_3d.py`). The assignment is detached from gradients.

The matcher chooses which queries receive GT class and box targets (`PETRHead._get_target_single`). A matched query is pushed toward its GT class and regressed toward its assigned GT. Every unmatched query receives background classification targets and zero regression weight (`PETRHead.loss_single`). Thus an added score loss can disagree with PETR's existing class loss; it cannot directly change a query's regression target. Its proposed route to better box supervision is to alter scores enough that future Hungarian matches change.

## Inputs and geometric measurements

For each training frame and decoder layer, use all 900 predicted boxes and logits, all GT boxes and classes, and camera calibration. Treat each (query q, GT g) as a candidate relationship before using its confidence or current Hungarian assignment.

For every valid camera view v, project the predicted and GT 3D boxes to image-clipped rectangles and calculate their 2D overlap. For this candidate relationship, choose the view with the **largest predicted-box–GT-box projected overlap**. Keep the chosen camera and exact GT together for every subsequent measurement. Invalid projections and tiny visible rectangles cannot establish a reliable relationship. Maximum projected overlap is geometric evidence; it does not prove which image features the query used.

Let o be the selected camera center, x_q the predicted 3D center, and x_g the GT center, expressed in the same coordinate system. In the horizontal plane define

\[
 u_g=\frac{x_{g,xy}-o_{xy}}{\lVert x_{g,xy}-o_{xy}\rVert},\quad
 e^r_{qg}=(x_{q,xy}-x_{g,xy})^T u_g,\quad
 e^t_{qg}=(x_{q,xy}-x_{g,xy})^T(-u_{g,y},u_{g,x}).
\]

The signed radial error e^r measures displacement along that camera-to-GT ray; e^t measures transverse displacement. Both are needed: horizontal center error satisfies d_xy² = (e^r)² + (e^t)². Record projected overlap, radial and transverse errors, vertical error, 3D/BEV box agreement, and all competing GTs. The selected camera can differ across query–GT pairs.

## Rules: a query role is decided before its penalty

The following are **operational rule labels**, not verified physical identities. A geometric case receives a hard loss target only when the evidence is sufficiently clear; otherwise it is excluded from the added loss.

| Role for class c | Rule decision | Added target |
|---|---|---:|
| Preferred candidate for GT g | Projection and ray geometry support g; g is a clear owner relative to **all** nearby GTs; this query has the best localization among credible candidates for g. | Continuous quality Q(q,g) |
| Inferior candidate for the same g | The query has a clear owner g, another credible query is better localized for g, and the quality gap is meaningful. It has no plausible competing GT owner. | 0 for c_g |
| Radial-error candidate | The query projects toward g but has a substantial along-ray displacement. If a better same-object candidate exists and ownership is clear, it can join the inferior group. If it is the only credible candidate, retain a quality target rather than automatically setting zero. | 0 or Q, according to the preceding condition |
| Clear class error or unsupported candidate | Evidence supports a different GT class or a genuinely unsupported region, with no plausible owner of class c. Mere lack of projected overlap is insufficient. | 0 for the incorrect class c |
| Competing-owner or uncertain candidate | The query plausibly belongs to another nearby GT, the projected evidence is weak, or two candidates are effectively tied. | No added target |
| GT without a credible candidate | None of the existing queries passes the ownership criteria. | No invented positive |

Candidate ownership is determined jointly across the frame. A query cannot be a preferred candidate for two GTs. Current class score and Hungarian assignment are recorded **after** the geometric role is decided; they cannot make a dubious query geometrically correct. Exact overlap, depth, ownership-gap, and tie cutoffs need calibration on PETR-R50 training queries and reviewed examples. These have not been established by the saved validation detections.

The first implementation assigns preferred and inferior targets for **all valid GT classes**. It does not yet assign zero targets to unsupported regions or to queries claimed by another GT class; those cases remain excluded. Every class uses the same provisional ray-quality scales in metres. Class-specific scales and evidence cutoffs may be needed for different object sizes and should be checked before interpreting AP changes.

For a credible candidate, define continuous geometric quality

\[
 Q(q,g)=\exp\!\left[-\left(\frac{|e^r_{qg}|}{\sigma_r}\right)^p
                       -\left(\frac{|e^t_{qg}|}{\sigma_t}\right)^a\right],
 \qquad \sigma_r,\sigma_t>0,\quad p,a>0.
\]

Small errors give a target near one; large errors give a target near zero. Scales are in metres. Higher exponents make the falloff sharper **around their scales**; they do not increase penalties uniformly. If the two scales and exponents are equal, Q is an isotropic XY-distance function and ray direction does not affect quality. We should choose anisotropy only if training data supports a distinct radial failure mode. Projection and cross-GT competition govern ownership; Q alone cannot tell whether a prediction belongs to the right person. Q is detached when used as a score target.

## Connect roles, HM assignment, and score to one loss

For each eligible (frame, layer, query, class) entry i, store the rule target t_i, current logit s_i, probability p_i = sigmoid(s_i), and Hungarian-assigned GT H_i. For a preferred query, t_i = Q(q,g). For a clear inferior or unsupported class prediction, t_i = 0. Ambiguous entries have no added target.

The Hungarian disagreement indicator must use the **assigned GT identity**, not only positive/background status:

- Preferred q for g but H_i is background or another GT: HM disagrees with the rule's owner. A same-class assignment to a different GT is especially important to log because class-score supervision cannot correct its regression owner directly.
- Clear inferior q for g but HM assigns q to g: HM and the rule give opposing class-score signals.
- Clear inferior q assigned to another credible GT: reconsider ownership; do not automatically suppress it.

**Batch score-rank rule.** For decoder layer ℓ and target class c, flatten the class-c score from *every query in every frame in the current per-device optimizer batch*. Define the one-based rank of query i as `R_i = 1 + count(j: score_j > score_i)` (ties are ignored for comparisons). A rank violation occurs when two reliable entries i,j for class c have a meaningful target-quality gap `t_i > t_j + δ` but `R_i > R_j`: the poorer entry currently scores above the better one. The implementation computes each eligible entry's fraction of such violations and increases its loss weight. Entries with uncertain geometric roles have no quality order and are excluded from violation pairs, although their scores still occupy positions in the actual batch rank. The selected base config has `samples_per_gpu=4`, so cross-frame comparisons occur among four frames on each GPU. They do not cross GPUs and are not the dataset-wide nuScenes AP order.

Define a bounded, detached weight w_i from (1) geometric-label reliability, (2) HM disagreement, and (3) ranking disagreement. For example,

\[
 w_i=a_i\,\operatorname{clip}(1+\beta h_i+\eta r_i,\,1,\,w_{max}),
\]

where a_i is zero for ambiguous cases; h_i is the HM disagreement and r_i is the fraction of comparable eligible entries whose score order contradicts their quality order. Beta and eta are nonnegative. This is the implemented weighting form, not a claim that any particular weights work. Keep the denominator independent of w_i so increasing a conflict weight actually increases its gradient. Normalize preferred and negative groups separately so numerous negative queries do not overwhelm the preferred queries. Ranking disagreement increases the soft-target score-loss weight. The pair term below also provides a direct ranking gradient.

The one added scalar term combines weighted soft-target binary cross-entropy and a within-batch pairwise ranking penalty over reliable entries of the same class and decoder layer:

\[
 L_{rule}=\sum_{\ell=1}^{6}\pi_\ell\left[
 \sum_{k\in\{preferred,negative\}}\rho_k
 \frac{1}{\max(1,|E_{\ell k}|)}
 \sum_{i\in E_{\ell k}}w_i
 \left[\operatorname{softplus}(s_i)-t_i s_i\right]
 +\mu\frac{1}{\max(1,|P_\ell|)}\sum_{(i,j)\in P_\ell}
 \operatorname{softplus}(m-s_i+s_j)\right],
\]

where `P_l` contains ordered pairs of the **same class** with `t_i > t_j + δ`, both roles reliable; m is a logit margin and μ is its weight. The implemented loss separately normalizes each active class's preferred and inferior groups, then averages over the active classes in that layer. Classes without a credible candidate contribute zero. Empty groups and pairs contribute zero; pi_l and rho_k are declared layer/group weights. The training objective becomes

\[
 L_{total}=L_{PETR,cls}+L_{PETR,reg}+\lambda L_{rule}.
\]

Roles, t_i, and w_i are detached. The BCE derivative with respect to an eligible logit, apart from the positive loss scaling, is

\[
 \frac{\partial L_{rule}}{\partial s_i}\propto w_i(p_i-t_i).
\]

Therefore a score below its target receives an upward BCE correction, and a score above its target receives a downward one. A preferred HM-negative query with **t_i ≤ p_i** receives no upward BCE correction even if HM disagrees; HM disagreement weighting cannot reverse that sign. The pair term supplies an upward ranking gradient if a lower-quality reliable query outscores it. If there is no such pair, the conflict remains unresolved. For an HM-positive inferior query, PETR's focal loss pushes its score up while the rule loss tends to push it down. The effective direction depends on the **sum** of their scaled gradients, so penalty weights must be checked against the existing focal loss rather than chosen from reward magnitudes alone.

## Intended mechanism and evidence required

A successful sequence would be: the rules identify a better candidate A for GT g; the added loss increases A's class score and lowers clear inferior candidates; a later Hungarian pass selects A; PETR's unchanged regression loss then refines A. Score ordering could improve before an HM switch, but improved localization through regression requires appropriate assignments. Other GTs can change the globally optimal match, so higher score alone does not guarantee A becomes positive.

The first audit should produce one row per eligible or excluded query-class entry with its role, candidate/competing GTs, camera, projected overlap, radial and transverse errors, Q, target, HM owner, score, batch/within-frame ranks, disagreement weights, and **combined baseline-plus-rule logit gradient**. Track the same summary by decoder layer. The current code returns preferred/inferior counts, HM conflicts, and rank violations from the loss helper but does not yet persist this full audit. Review a score-stratified sample of proposed duplicates, cross-GT cases, and unsupported candidates in images. Then compare continued baseline fine-tuning with the one-term proposal on scene-disjoint training subdivisions and measure assignment switches, score ordering, geometric quality, and official AP at every nuScenes threshold on held-out scenes.

Existing PETR-R50 train-subset detections under `experiments/nuscnes_exp/train_split_diagnostics/outputs/petr_r50/` show that Rule 4 alternatives occur in 29,939 of 116,310 matched pedestrian GTs (25.74%) across 25,433 available train frames. The source token audit records 2,697 missing train frames. These final detections motivate the study; they do not provide raw 900-query HM status, validated query roles, or evidence that this loss improves AP.

The unresolved scientific choices are the ownership and ambiguity criteria, whether a representative's radial quality should be anisotropic, whether a soft quality target can overcome HM-background supervision, and the weight needed to change the combined gradient without damaging other classes. The audit should resolve those choices before training claims are made.
