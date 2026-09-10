# Official DeepPro metric amendment (2026-09-10)

This amendment supersedes the model-selection and C3-continuation metric rules
in `SELECTION_AND_EXACT_LOGIT_PLAN_2026-09-09.md`.

The user directed the project to stop using F1 as a detection metric and to
align evaluation with `git@github.com:TinaLRJ/DeepPro.git`. The directive was
recorded after all three B1 runs had completed, but before any C0, C1, or C2
candidate had completed training or produced a metric payload. Existing B1
payloads remain valid because their `all.pd` and `all.fa` are measured at
sigmoid threshold 0.5 and their `all.auc` was already recomputed from the
official 27-threshold subset.

## Detection metric contract

Only the following quantities may enter a continuation gate, candidate
qualification, ranking, lock, or headline result:

1. target-level Pd at sigmoid threshold 0.5;
2. pixel-level Fa at sigmoid threshold 0.5;
3. area under the Pd-Fa curve evaluated on the official 27 thresholds.

The 27 thresholds are `0, 1e-20, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4,
1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1`.

Training loss and training IoU may be reported only as optimization
diagnostics. Legacy pixel metrics already stored in raw payloads or logs are
ignored by active analyzers and selectors. Dense-grid AUC, fixed-budget
workpoints, and raw-logit scans are optional sensitivity diagnostics and do not
authorize, qualify, rank, or lock a model.

## Candidate qualification and ranking

A candidate qualifies only if all of these conditions hold across seeds
47/49/51:

1. mean 27-threshold AUC is not below the paired B1 mean;
2. mean Pd@0.5 is no more than one percentage point below B1;
3. mean Fa@0.5 relative reduction versus B1 is positive;
4. at least two of three seeds have both Pd@0.5 not below B1 and Fa@0.5 not
   above B1;
5. every seed has a validation latency ratio at most 1.3;
6. split, profile, checkpoint, FP32, scratch-only, and run identity checks pass.

Qualified candidates are compared jointly by mean Pd@0.5 (higher), mean
Fa@0.5 (lower), and mean 27-threshold AUC (higher). A candidate can be locked
automatically only when it is the sole Pareto-nondominated qualified model:
it must be no worse than each competing qualified candidate on all three
metrics and strictly better on at least one. Multiple nondominated candidates
remain an unresolved trade-off and fail closed; no scalar weighting,
single-metric ordering, or parameter-count tie-break is used. If no candidate
qualifies, B1 is locked.

The C3 continuation decision uses the same official Pd@0.5, Fa@0.5, and
27-threshold AUC evidence. It no longer requires raw-logit agreement.

Official test isolation, the fixed 64/16 internal split, FP32 execution,
scratch-only initialization, seeds, epoch 32, and GPU 0/1/2 policy are
unchanged. No content hash is generated or checked.
