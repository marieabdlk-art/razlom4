# HDC Cognitive Prototype — experiment result

## Verdict

```text
VALUABLE_HDC_CORE
DYNAMIC_SEMANTIC_OVERLAY_UNPROVEN
```

The bounded HDC associative-memory core passed the preregistered engineering
value gate on ten previously uninspected final seeds. The additional mutable
semantic overlay did not establish an incremental advantage over static HDC and
should remain experimental rather than the production default.

## Protocol

- Development seeds: 1–5.
- Frozen final seeds: 101–110.
- Equal event and declared context-capacity budget.
- Controls: exact 4-gram, backoff n-gram, sparse character n-gram KNN, static
  HDC, and dynamic HDC.
- Tasks: clean held-out composition, corrupted contexts, ordered-role probes,
  continual retention/acquisition, loop resistance, replay, rollback, capacity,
  and outcome overflow.
- Parameters were frozen before final seeds were run: five-neighbor voting,
  stable-key weight 0.8, semantic drift cap 0.01 per update.

The first draft of the order task accidentally used an unlearnable random
target rule. It was rejected before parameter selection, replaced by a declared
ordered relation, and never used as positive evidence. Preserved result files
begin with the corrected oracle.

## Final-seed results

| Arm | Clean accuracy | Noisy accuracy | Ordered-role accuracy | Retention/acquisition harmonic | Logical bytes |
|---|---:|---:|---:|---:|---:|
| Exact 4-gram | 28.88% | 0.00% | 0.00% | 0.00% | 29,490 |
| Backoff n-gram | **43.80%** | 35.45% | 0.00% | 0.00% | **39,098** |
| Character KNN | 32.73% | 37.55% | 42.00% | 33.55% | 222,785 |
| Static HDC | 43.75% | **43.23%** | 85.00% | **59.21%** | 89,982 |
| Dynamic HDC | 43.58% | **43.23%** | **88.00%** | 59.15% | 404,427 |

Additional hard-gate results:

- mean retention drop after stream B: **0.00 percentage points**;
- loop failure rate across 2,000 generations: **0.00%**;
- clean regression of dynamic HDC versus strongest non-HDC arm: **0.23 points**;
- noisy-context advantage versus strongest non-HDC arm: **+5.68 points**;
- all 44 repository tests passed;
- exact behavior-hash replay and rollback passed.

## What is genuinely valuable

Static HDC provides the best current engineering tradeoff:

- essentially matches backoff n-gram accuracy on clean held-out combinations;
- is 5.68 points better than the strongest non-HDC arm on corrupted contexts;
- captures ordered relations that the n-gram controls do not transfer;
- retains old stream behavior while acquiring a second task;
- uses less than half the logical context-memory bytes of sparse character KNN;
- remains bounded, deterministic, rollback-safe, and API-free.

This supports a narrow use case: compact, noise-tolerant, online associative
memory for small symbolic/text streams.

## What is not established

- Dynamic semantics used about 4.5 times the logical memory of static HDC.
- Its final noisy accuracy was identical to static HDC.
- Its ordered-role score was only three points higher and its retention harmonic
  was slightly lower.
- Hormones and coherence are operational diagnostics, not demonstrated sources
  of accuracy.
- Synthetic next-token tasks do not establish natural-language understanding,
  cognition, consciousness, or AGI.
- The benchmark is still small and generated; real corpora, stronger learned
  baselines, profiling, and independent reproduction remain necessary.

## Reproducibility artifacts

- [`HDC_BENCHMARK_PLAN.md`](HDC_BENCHMARK_PLAN.md)
- [`hdc_benchmark.py`](hdc_benchmark.py)
- [`runs/hdc_dev_selected.json`](runs/hdc_dev_selected.json)
- [`runs/hdc_final_test.json`](runs/hdc_final_test.json)
- [`test_hdc_benchmark.py`](test_hdc_benchmark.py)
- [`test_hdc_cognitive.py`](test_hdc_cognitive.py)

The correct product default is static HDC. Dynamic semantics remains an
explicit opt-in research mode until it beats the static ablation under the same
memory budget.
