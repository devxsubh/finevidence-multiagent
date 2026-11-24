# The research loop: agents that improve FinAgents

A second LangGraph workflow (`src/finagents/research/loop.py`) runs a team of seven agents that experiment on the question-answering system itself.

```
analyse performance → identify weakness → generate hypothesis ─(protected)─▶ request approval ─┐
        ▲                                        │                                             │
        │                                  design experiment → implement → run → evaluate       │
        │                                        │                                             │
        │                                  research critic review                               │
        │                                   │ valid                │ not valid                 │
        │                             save candidate          analyse failure                   │
        │                                   └──────────┬───────────┘                           │
        └──────────────── next iteration ◀──────── document ◀──────────────────────────────────┘
```

| Agent | Job |
| --- | --- |
| Research analyst | reads the verified facts in `docs/background/research-notes.md` (it does not generate claims) and the current results; ranks weaknesses by how many questions they affect |
| Dataset agent | profiles train questions: operation types, evidence source (table, text, both), gold-fact counts, program lengths |
| Experiment designer | maps the top weakness to an untested hypothesis; fixes the control, the slices and the acceptance criterion before anything runs |
| ML engineer | applies the hypothesis to the control's settings and validates the result as a `SystemConfig`; saves accepted candidates to `configs/candidates/` |
| Evaluation agent | runs control and variant on the same 600 selection and 600 confirmation train questions |
| Research critic | protocol and leakage checks; exact McNemar test on the objective (p < 0.05, improving); the guard metric may not worsen by more than 1 point; the confirmation slice must move the same way |
| Documentation agent | appends each iteration to `docs/research-log.md` and `runs/research/<session>.json` |

## What the agents may and may not change

Experiments are drawn from a fixed registry (`research/registry.py`): each hypothesis is a bounded change to tunable settings (confidence threshold, search depth, research rounds, relevance threshold, ambiguity margin, period penalty, unit strictness, LLM use). Data, splits and metrics are not tunable, and the critic rejects any change outside the tunable set.

**Approval gates.** These never happen without a person:

| Action | Why |
| --- | --- |
| Run a protected hypothesis (H9: let Claude reason) | external dependency and per-question API cost |
| Evaluate on the official test split | the single final evaluation is a human decision |
| Promote a candidate to the default configuration | changes the system everyone uses |
| Change the benchmark, metric or split protocol | otherwise the loop could optimise its own evaluation |

The loop files an approval request (`runs/research/approvals.json`) and moves on to the next weakness. Approve or reject with `uv run finagents research approve <id>` or in the frontend's Research lab; an approved hypothesis runs in the next session.

## First session (R20261124-163148, 4 iterations)

| # | Hypothesis | Outcome |
| --- | --- | --- |
| 1 | H1 raise `min_confidence` 0.60 → 0.70 | **accepted**: wrong-answer rate −4.0 pts (25 vs 1 questions, p = 8e-7), accuracy −0.7 pts, confirmed (−4.3 pts) on the second slice |
| 2 | H5 stricter relevance | rejected: 1 better vs 3 worse, p = 0.63 |
| 3 | H6 wider ambiguity margin | rejected: guard breached (accuracy −1.0 pts) |
| 4 | H8 relaxed units | rejected: no change |

On dev, the accepted candidate behaves as it did on train: wrong-answer rate 28.3% → 24.8%, accuracy when answering 39.9% → 42.5%, execution accuracy 18.8% → 18.4%. It has **not** been promoted to `configs/full.toml`; that is a human decision.

## Second session (R20261124-164250, 7 iterations, started from the Research lab page)

It repeated the first session's findings (H1 accepted again; the other configuration hypotheses rejected) and then reached **H9 (Claude reasoning)**. Because H9 is protected, the loop did not run it: it filed approval request `89cab4c7` and continued. The request is still pending; approving it only has an effect with an `ANTHROPIC_API_KEY` set.
