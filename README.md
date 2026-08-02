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

## KUDS → RAZLOM-4 full pipeline

Version 0.2 adds an optional generation layer before the conflict protocol:

```text
task contract
  → KUDS baseline snapshot + 12 divergent seeds
  → deterministic novelty/diversity selection of 4 seeds
  → 4 blind role commits
  → conflict + causal mutation
  → anonymous non-self review
  → hard gates + maximin
  → idea dossier or failure certificate
```

KUDS expands the search space. RAZLOM-4 then tries to break and mutate the
selected ideas. The combination does not weaken the original protocol: roles
remain blind during commit, authors still discard a frozen assumption, and the
existing deterministic validator performs final selection.

The full run uses 13 model calls: one KUDS divergence call plus the original
three rounds of four calls. The final artifact includes the complete KUDS pool,
seed-selection scores, protocol session, deterministic selection result, and a
compact idea dossier. Unexecuted experiments remain explicitly
`NOVELTY_UNCONFIRMED`.

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

Run the combined pipeline through OpenRouter:

```bash
export OPENROUTER_API_KEY="..."
razlom4-full full_pipeline_task.json \
  --output runs/support-routing.json \
  --model z-ai/glm-5.1
```

For reproducible offline runs, pass an ordered response artifact instead of an
API key:

```bash
razlom4-full full_pipeline_task.json \
  --output runs/replayed.json \
  --candidates 4 \
  --replay responses.json
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
- [`razlom4_full.py`](razlom4_full.py) — KUDS generation and full orchestration.
- [`full_pipeline_task.json`](full_pipeline_task.json) — example task contract.
- [`RAZLOM4_KUDS_GLM51_Colab.ipynb`](RAZLOM4_KUDS_GLM51_Colab.ipynb) — ready-to-run GLM-5.1 Colab notebook.
- [`HDC-architecture-spec-v3.md`](HDC-architecture-spec-v3.md) — implementable reference specification produced by the HDC case study.
- [`hdc_cognitive.py`](hdc_cognitive.py) and [`HDC_COGNITIVE_V3_Colab.ipynb`](HDC_COGNITIVE_V3_Colab.ipynb) — API-free runnable HDC core and Colab benchmark against an n-gram baseline ([open in Colab](https://colab.research.google.com/github/marieabdlk-art/razlom4/blob/main/HDC_COGNITIVE_V3_Colab.ipynb)).
- [`HDC_EXPERIMENT_RESULT.md`](HDC_EXPERIMENT_RESULT.md) — preregistered ten-seed evidence for a narrow noise-tolerant HDC-memory use case, plus the negative ablation result for dynamic semantics.
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
