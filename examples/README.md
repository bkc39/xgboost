# Examples

Runnable, self-contained programs that exercise the `xgboost` Racket bindings.
Each file embeds or synthesizes its own data — no downloads — and most assert
their own behavior with RackUnit so they double as end-to-end tests.

## Running

From the repo root, inside the Nix dev shell:

```bash
racket examples/01-train-regression.rkt        # run one example
raco test examples/27-get-started.rkt          # run its RackUnit checks
```

The fast, assertion-backed subset runs as part of `nix build`; see `AGENTS.md`
for the exact list. The longer narrative demos (the real-dataset ones) stay
manually runnable. The CUDA examples (24, 25) need a physical NVIDIA GPU and a
CUDA-enabled native build; they skip gracefully on CPU-only builds.

## Start here

If you are new to the bindings, read the
[user guide](../xgboost/scribblings/xgboost-guide.scrbl) (published at
<https://docs.racket-lang.org/xgboost/>) and run **`27-get-started.rkt`**, the
Racket counterpart of XGBoost's
[Get Started](https://xgboost.readthedocs.io/en/stable/get_started.html) page.

## Index

### Quickstart & core tasks

| File | Shows | Upstream analogue |
| --- | --- | --- |
| `27-get-started.rkt` | LIBSVM file → train → predict → save/load round-trip | [Get Started](https://xgboost.readthedocs.io/en/stable/get_started.html) |
| `00-print-dmatrix.rkt` | Smallest end-to-end exercise of the DMatrix primitives | — |
| `01-train-regression.rkt` | End-to-end regression training run | `demo/guide-python` |
| `02-train-classifier.rkt` | End-to-end binary-classification training run | `demo/guide-python` |
| `04-train-multiclass.rkt` | Three-class classification on a synthetic dataset | `demo/multiclass_classification` |
| `05-train-with-eval.rkt` | Training while watching loss on a held-out eval set | `demo/guide-python` (eval watchlist) |
| `06-iris.rkt` | Batteries-included Iris classification (embedded CSV) | `demo/multiclass_classification` |

### Objectives (built-in and custom)

| File | Shows | Upstream analogue |
| --- | --- | --- |
| `23-custom-objective.rkt` | Custom squared-error objective, two equivalent ways | [Custom Objective](https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html), `demo/guide-python/custom_objective.py` |
| `07-robust-regression.rkt` | Squared error vs Huber vs L1 (absolute error) | Custom objectives |
| `08-quantile-regression.rkt` | Quantile regression — three quantiles on Diabetes | Quantile regression |
| `09-poisson-bikes.rkt` | Poisson count regression on UCI Bike Sharing | `count:poisson` |
| `10-aft-survival.rkt` | `survival:aft` on Veterans' lung-cancer data | [Survival Analysis (AFT)](https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html), `demo/aft_survival` |

### Ranking

| File | Shows | Upstream analogue |
| --- | --- | --- |
| `29-learning-to-rank.rkt` | LambdaMART `rank:ndcg` with query groups; asserts high held-out nDCG | [Learning to Rank](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html) |

### Parameter recipes

| File | Shows | Upstream analogue |
| --- | --- | --- |
| `28-param-recipes.rkt` | DART, monotonic constraints, interaction constraints, random-forest mode | [DART](https://xgboost.readthedocs.io/en/stable/tutorials/dart.html), [Monotonic](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html), [Interaction](https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html), [Random Forests](https://xgboost.readthedocs.io/en/stable/tutorials/rf.html) |

### Model IO & persistence

| File | Shows | Upstream analogue |
| --- | --- | --- |
| `03-save-load.rkt` | Persist to a file and to an in-memory bytes blob | [Introduction to Model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html) |
| `26-booster-snapshot.rkt` | `booster->bytes` / `bytes->booster` full-state snapshot | Model IO (checkpoint/resume) |

### DMatrix construction & metadata

| File | Shows | Upstream analogue |
| --- | --- | --- |
| `12-dmatrix-constructors.rkt` | Every high-level DMatrix constructor (dense/CSR/CSC/columnar/URI) | [Text Input Format](https://xgboost.readthedocs.io/en/stable/tutorials/input_format.html) |
| `14-dmatrix-metadata.rkt` | Label/weight/group/feature-info round-trips | — |
| `15-dmatrix-slicing-binary.rkt` | Row slicing and binary DMatrix serialization | — |
| `16-quantile-cuts.rkt` | Quantile-cut inspection (`hist` tree method) | — |

### Booster inspection & lifecycle

| File | Shows | Upstream analogue |
| --- | --- | --- |
| `13-high-level-root-api.rkt` | Smoke test exercising the high-level root API end to end | — |
| `17-booster-lifecycle-config.rkt` | Reset / slice / boosted-rounds + JSON config round-trip | [Slicing Models](https://xgboost.readthedocs.io/en/stable/tutorials/slicing_model.html) |
| `18-booster-attrs.rkt` | Booster string attributes | — |
| `19-booster-dumps-feature-scores.rkt` | Model dumps (text/json/dot) and feature importance | Plotting / feature importance |

### Serving: in-place prediction

| File | Shows | Upstream analogue |
| --- | --- | --- |
| `20-inplace-predict-dense.rkt` | In-place prediction on a dense row-major matrix | `inplace_predict` |
| `21-inplace-predict-csr.rkt` | In-place prediction on a CSR sparse matrix | `inplace_predict` |
| `22-inplace-predict-columnar.rkt` | In-place prediction on columnar (struct-of-arrays) input | `inplace_predict` |

### Global / process APIs

| File | Shows | Upstream analogue |
| --- | --- | --- |
| `11-global-apis.rkt` | Version, build-info, and process-global config round-trips | — |

### GPU (requires NVIDIA GPU + CUDA build)

| File | Shows | Upstream analogue |
| --- | --- | --- |
| `24-cuda-regression.rkt` | CUDA-accelerated regression training | GPU support |
| `25-cuda-classification.rkt` | CUDA-accelerated binary classification training | GPU support |

## Conventions

- Assertion-backed examples `(provide run-example)`, print concise output from
  `module+ main`, and verify behavior from `module+ test`.
- New public API or user-visible features should land with an example-backed
  end-to-end test in the same change set (see `AGENTS.md`).
