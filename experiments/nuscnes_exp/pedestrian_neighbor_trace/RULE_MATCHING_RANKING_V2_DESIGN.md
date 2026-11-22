# Rule assignment and ranking, V2 experiment

This is an opt-in one-epoch pilot starting from the same PETR-R50 epoch-24
checkpoint as V4/V5. V6 keeps PETR's original matching and adds the new ranking loss. V7
uses rule-guided one-to-one matching and **the identical ranking loss**. All
legacy configs retain the old behavior.

`rule_v2_cfg=None` disables both new components. Within an enabled V2 config,
`matching='original'` or `'rule'` selects the target assignment, and
`enable_ranking` independently turns the additional score loss on or off.

## Assignment contract (each decoder layer and frame)

1. Inspect every query's complete sigmoid class-score vector. Pair (query q,
   GT g of class c) is class compatible only if p(q,c) exceeds a permissive
   floor and its logit is within a configured margin of q's largest class
   logit. For example, car 0.01 versus pedestrian 0.4 must not certify a car
   pair. Scores gate eligibility, but do not pick the winner among eligible
   boxes. Numerical gates are initial hypotheses to audit on train data.
2. Project predicted and GT boxes to all valid cameras. Maximum projected
   rectangle IoU is the primary owner evidence. A candidate is certified only
   when its best GT overlap exceeds a configured floor, beats the next-best GT
   by a margin, and the query has plausible ground-plane center distance.
   This prevents a strongly projected but grossly wrong-depth box from being
   certified. The best-overlap camera supplies a camera ray; center error is
   decomposed into signed radial and tangential components. Both contribute
   to ground-plane center distance sqrt(radial^2 + tangential^2).
3. For each GT, select the certified query with smallest ground-plane center
   error. Projected overlap establishes ownership; better 3D localization
   chooses the representative. A query has at most one clear projected owner,
   and global assignment reserves the certified representatives first.
4. For GTs without a certified representative, use **the original PETR
   classification-plus-regression cost** on the remaining GTs and unclaimed
   queries. Hungarian matching solves this fallback subproblem globally. Its
   matches receive ordinary PETR classification and box targets, but are
   labelled `hm_fallback` and are not automatically ranking positives. This
   handles zero overlap, out-of-camera GTs, and low-score objects without
   forcing a rule-certified TP. The original all-query/all-GT assignment is
   computed independently for comparison.
5. Among nonselected queries, a class-compatible query with the same clear GT
   owner and meaningfully worse center localization than a certified winner
   is a confirmed duplicate, including good-projection/wrong-depth cases.
   Do not label a query negative for class c if it may credibly represent a
   different GT of class c. A query projecting clearly to a different-class
   GT can be negative for class c when no class-c GT supports it. A query with
   no GT support and large center error from every GT may also be negative
   for class c. Near ties and
   ambiguous ownership receive no new label. Original HM background alone
   is never a reason to mark a query rule-negative.

## One added score-ranking loss

For class c, define P_c as certified GT representatives and N_c as confirmed
duplicate or unsupported negatives. Fallback matches are excluded from P_c
unless they separately meet certification criteria. Use logits z(q,c):

    L_rank(c) = a mean_{i in P_c} softplus(-z(i,c))
              + b mean_{j in N_c} softplus( z(j,c))
              + d mean_{i in P_c, j in hard(N_c)}
                    softplus(m + z(j,c) - z(i,c))

Average over active classes and decoder layers, then multiply by the opt-in
outer weight. The pair term raises a valid representative and lowers a
confirmed FP, including across frames in one GPU batch. It never compares
two GT representatives as positive versus negative. Hard negatives are the
highest-scoring confirmed negatives, bounded per class to control memory.
Absolute class-score anchors retain a gradient when no informative pairs
exist. Localization chooses roles; it is not a soft confidence target.

This score loss has no direct box-regression gradient. In V6, PETR's original
matching may oppose its score gradient. In V7, rule-certified matches receive
ordinary PETR classification and regression supervision, while fallback
matches retain the original PETR cost. The geometry and role selection are
detached in both runs. A fixed ranking weight does not guarantee the combined
logit gradient direction; measure it during training.

## Comparison and acceptance audit

Log per-layer certified count, HM fallback count, original-versus-rule match
changes, center-distance improvement/worsening on switched GTs, and class
compatibility rejection count. Each epoch also saves a bounded rank-0 sample
of switched GT/query pairs with class scores, projected overlaps, and radial,
tangential and center errors; this sample is illustrative, while count totals
are reduced across GPUs. Compare V6 and V7 from the same checkpoint,
schedule, freeze policy and ranking weight. On train frames, inspect switched
pairs with low/zero overlap, nearby competing GTs, and wrong-depth duplicates
before interpreting AP. Validate per-class AP at 0.5/1/2/4 m, top-300 survival,
and global class-score order. A detection can be valid at 4 m and invalid at
2 m; assignment labels are a training approximation rather than official
threshold-independent TP/FP truth.

Before fine-tuning, run `qsub tools/audit_nuscenes_petr_r50_rule_v2_assignments_a100.pbs`.
It uses the original checkpoint and training images without optimizer updates,
writing `summary.json` and `switched_pairs.jsonl` to
`results/petr_r50dcn_gridmask_p4_rule_v2_assignment_audit/`. The initial
20 batches are a smoke audit, not a representative estimate of all train
scenes; increase `MAX_BATCHES` for a broader audit after reviewing examples.
