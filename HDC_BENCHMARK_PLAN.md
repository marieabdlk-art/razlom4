# HDC Cognitive Prototype — preregistered benchmark plan

Status: frozen before prototype iteration 2.

## Objective

Determine whether the runnable HDC system provides a measurable online-memory
advantage rather than merely replaying text. The benchmark must also reject
systems that gain accuracy by looping, forgetting old knowledge, hiding
capacity failures, or losing deterministic rollback.

## Frozen data split

- Development seeds: `1, 2, 3, 4, 5`.
- Final test seeds: `101, 102, 103, 104, 105, 106, 107, 108, 109, 110`.
- Test-seed aggregate is inspected only after an iteration is selected using
  development seeds.

## Equal-budget arms

1. Exact bounded 4-gram.
2. Bounded backoff n-gram.
3. Bounded character-n-gram nearest-neighbor memory.
4. Static HDC (semantic overlay disabled).
5. Dynamic HDC.

All arms receive the same train events and a declared context-record capacity.
Memory accounting includes stored contexts, outcomes, semantic accumulators,
and indexes. Python object overhead is reported separately from logical bytes.

## Tasks

### T1 — Compositional held-out prediction

Train and test use the same grammar but disjoint complete sentences and novel
subject/verb/object/place combinations. Measure next-token accuracy and
coverage.

### T2 — Noisy-context robustness

Train on clean sequences. Test contexts receive one deterministic character
deletion, adjacent transposition, or keyboard-independent substitution while
the target remains clean. Report accuracy by corruption type.

### T3 — Order/role sensitivity

Use minimal pairs with the same tokens but reversed actor/recipient or object
order. A prediction is correct only when it follows the ordered relation.

### T4 — Continual-learning retention

Learn stream A, freeze A probes, then learn a conflicting stream B sharing
symbols. Report A accuracy before/after, B acquisition, and harmonic mean.

### T5 — Loop resistance

Train on repetitive and branching corpora, then generate from 200 prompts.
Failure means a repeated token 4-gram appears more than twice or generation
exceeds its declared step bound without an explicit stop reason.

### T6 — State integrity

Check 100 exact replays, consolidation crash-safe semantics at the Python state
level, full behavior-hash rollback, capacity refusal, and the ninth outcome.

## Metrics

- accuracy, coverage, and selective accuracy;
- paired per-seed score differences;
- A-retention drop and A/B harmonic mean;
- loop-failure rate and useful generated length;
- logical memory bytes and records;
- train/query wall time as descriptive metrics only;
- replay/rollback equality.

## Hard gates

1. All unit and repository tests pass.
2. Replay and rollback equality are 100%.
3. No hidden capacity growth or implicit deletion.
4. Loop-failure rate is at most 2%.
5. Retention drop is at most 5 percentage points, unless the system explicitly
   reports a contradiction and quarantines the update.

## Value gate

The prototype is empirically valuable if all hard gates pass and at least one
of the following is true on final seeds without a >2 point regression on clean
compositional accuracy:

- dynamic HDC beats the strongest equal-budget non-HDC arm by at least 3 mean
  percentage points on noisy-context accuracy; or
- dynamic HDC is within 1 point of the strongest arm while using at most half
  its logical context-memory bytes; or
- dynamic HDC beats static HDC by at least 3 points on the A/B retention-
  acquisition harmonic mean.

These are engineering evidence for a bounded associative-memory use case, not
evidence of language understanding or general intelligence.
