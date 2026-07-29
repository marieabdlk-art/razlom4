# RAZLOM-4

**RAZLOM-4 is a falsifiable multi-agent conflict protocol.** Four agents do not
vote, average their answers, or hand the decision to another LLM. They are
forced to transform disagreement into a testable causal mutation — or return a
failure certificate.

[Русская версия](README.ru.md)

## Why another debate protocol?

Typical multi-agent debate converges toward a polished compromise:

```text
proposals → discussion → LLM judge → synthesis
```

RAZLOM-4 is designed for decisions where compromise can hide the fatal flaw:

```text
task contract
  → 4 blind role commits
  → full cross-conflict + 4 mutations
  → anonymous non-self review
  → hard gates → maximin → result or failure certificate
```

The four roles optimize incompatible objectives:

| Role | Optimizes | Must expose |
|---|---|---|
| Architect | measurable effect and coverage | complexity cost |
| Skeptic | falsifiability and minimal assumptions | condition for conceding |
| Simplifier | fewer components and states | lost functionality |
| Failure Hunter | worst-case survival and reversibility | residual risk |

## What makes it different

- Proposals are committed before agents see one another.
- Each author must discard one of their own frozen assumptions.
- Each mutation must preserve a foreign invariant and contain pressure from at
  least three roles.
- Novelty means a changed state transformation, not new wording.
- Every candidate needs a prediction, falsifier, ablation, experiment, and
  rollback.
- A veto is valid only with a constraint, counterexample, kill-test, and
  observed failure.
- Authors never review their own candidate.
- Selection is deterministic: hard gates, then external maximin scores.
- `NO_VALID_MUTATION` is a valid result. The protocol never has to invent a
  winner.

The protocol can establish at most `panel_novel`: different from the four
frozen proposals. Claims of global novelty require a separate literature or
patent review.

## Quick start

RAZLOM-4 has no runtime dependencies and requires Python 3.9+.

```bash
python3 razlom4.py validate example_session.json
python3 razlom4.py select example_session.json
python3 -m unittest discover -v
```

Or install the local CLI:

```bash
python3 -m pip install .
razlom4 validate example_session.json
razlom4 select example_session.json
```

The example selects `uncertainty_reversibility_gate`: instead of compromising
between slow human review and unsafe automatic answers, it changes the object
of automation from *answer generation* to *routing by uncertainty and
reversibility*.

## Blind benchmark harness

The repository includes a deterministic scorer for five equal-budget arms:

1. single agent;
2. best-of-4;
3. ordinary debate;
4. RAZLOM-4 without incompatible roles;
5. full RAZLOM-4.

```bash
python3 razlom4_benchmark.py validate-bank \
  benchmark/public-task-bank.json benchmark/private-task-bank.json

python3 razlom4_benchmark.py evaluate \
  benchmark/public-task-bank.json \
  benchmark/private-task-bank.json \
  benchmark/manifest.json \
  benchmark/examples/smoke-submission.json
```

The bundled three-task bank is explicitly a smoke test. It validates sealing,
budget enforcement, hidden-oracle scoring, Wilson intervals, and paired sign
tests; it is not evidence that the method outperforms its controls.

## Honest current evidence

The first blind case study tested recovery from a partially applied,
irreversible cluster rollout. All four mutations received confirmed vetoes and
the only allowed repair returned `REJECT`. Final result:

```text
NO_VALID_MUTATION
```

A single-agent control also recognized the information-theoretic impossibility;
best-of-4 selected a conditional forward path that collapsed to `BLOCK`. The
case demonstrates useful hostile-audit behavior, but does **not** establish an
accuracy advantage over one strong agent. See
[`case-studies/partial-failure/RESULT.md`](case-studies/partial-failure/RESULT.md).

## Repository map

- [`METHOD.md`](METHOD.md) — normative protocol specification.
- [`PROMPTS.md`](PROMPTS.md) — role and round prompts.
- [`protocol.schema.json`](protocol.schema.json) — session artifact schema.
- [`razlom4.py`](razlom4.py) — validation, hard gates, and selection.
- [`example_session.json`](example_session.json) — reproducible example.
- [`benchmark/`](benchmark/) — blind scorer contracts and smoke bank.
- [`RAZLOM4_BENCHMARK_PLAN.md`](RAZLOM4_BENCHMARK_PLAN.md) — preregistered evaluation design.
- [`case-studies/`](case-studies/) — preserved positive and negative traces.

## Scope and limitations

RAZLOM-4 does not guarantee invention, truth, independence between agents from
the same model family, or superiority per token. It guarantees a bounded,
auditable process that separates proposal, conflict, mutation, testing, and
selection.

The main protocol uses 12 model calls: three rounds of four. It is intended for
high-conflict architecture, experiment-design, safety, and irreversible
decision problems — not routine questions where one strong agent and a test
suite are cheaper.

## License

[MIT](LICENSE)
