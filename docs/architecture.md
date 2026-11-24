# Architecture

## Principles

1. **LLMs decide what to do; deterministic tools decide what the numbers are.** The only way an agent can refer to a figure is by evidence id (`E1`). The reasoning agent writes `subtract(E2, E1), divide(#0, E1)`; ids are replaced by audited values; the restricted executor computes the result. A program in which a model wrote a number itself is rejected.
2. **Agents never see answers.** Every agent works from `inference_input(record)`: the question, table, pre_text and post_text. Answers, programs and gold evidence are only read by evaluation and by the reranker's training on the train split.
3. **Controlled execution, not free-form autonomy.** LangGraph runs a fixed graph with explicit transitions, typed state, conditional routing and retry budgets, so every run ends and every decision is logged.
4. **Every answer is auditable.** The result carries the plan, retrieved segments, audited evidence, the program, each execution step, every verification check and the trace of all agent steps with the orchestrator's routing decisions.

## The graph (`src/finagents/agents/orchestrator.py`)

```
START → orchestrator → query_analyst ─(unsupported)──────────────────────────────▶ abstain
                            │
                            ▼
                     evidence_researcher ◀──────────── research again (round + 1, wider k) ───┐
                            │                                                                │
                            ▼                                                                │
                     evidence_auditor ─(insufficient, rounds left)───────────────────────────┤
                            │ valid                                                          │
                            ▼                                                                │
                        reasoning ─(no candidate)────────────────────────────────────────────┤
                            │                                                                │
                            ▼                                                                │
           ┌──────▶ program_validator ─(invalid)─▶ next candidate                             │
           │                │ valid                                                          │
           │                ▼                                                                │
           │           calculation ─(error)─▶ next candidate                                 │
           │                │                                                                │
           │                ▼                                                                │
           └─(failed checks) verification ─(low confidence / ambiguous)──────────────────────┘
                            │ verified          └─(conflicting evidence)─▶ abstain
                            ▼
                        response ◀── abstain (retry budget spent)
                            │
                           END
```

The orchestrator wraps every agent node: after the agent returns, `Orchestrator.decide()` reads the merged state, chooses the next node, applies its own state changes (the research round counter, the candidate pointer) and records the decision in that step's trace entry. The budgets are `max_research_rounds` (default 3) and `max_candidates` (12), so the graph always terminates.

## State and messages (`src/finagents/state.py`)

`AgentState` is a `TypedDict`; every message in it is a Pydantic model, validated at each hand-off and serializable for the API.

| Field | Written by | Message |
| --- | --- | --- |
| `payload` | orchestrator | allowlisted question + report |
| `analysis` | query analyst | `QueryAnalysis`: metric, entity terms, periods, operation, required evidence |
| `retrieved`, `found` | evidence researcher | `RetrievedSegment`s; candidate `EvidenceValue`s |
| `audit` | evidence auditor | `AuditReport`: approved `EvidenceValue`s with ids, checks, status |
| `candidates`, `candidate_index` | reasoning agent / orchestrator | `CandidateProgram`s over evidence ids |
| `validation` | program validator | `ValidationReport` |
| `calculation` | calculation agent | `Calculation` with every step |
| `verification`, `attempts` | verification agent | `Verification` (checks, confidence, failure kind) |
| `abstain_reason`, `route`, `research_round` | orchestrator | routing state |
| `final` | response agent | `FinalAnswer` |
| `trace` | every node | `AgentStep` (append-only) |

## Modules

| Module | Responsibility |
| --- | --- |
| `core/records.py` | load splits (train and dev only), allowlisted inference input, gold fields for evaluation |
| `core/values.py` | parse "$ 1,234", "( 12 )", "4.3%" to `Decimal`; units; never rescales |
| `core/program.py` | FinQA program grammar; parse, never eval |
| `core/executor.py` | official FinQA execution rules; reproduces `exe_ans` for all 7,134 train + dev gold programs |
| `core/segments.py` | table rows and sentences keyed like FinQA gold evidence; value mentions with label, period, unit |
| `retrieval/` | BM25, TF-IDF (train IDF), learned reranker (MLP over 13 features, trained on train) |
| `llm/` | `LLMClient` protocol; `ClaudeClient` (structured outputs); `None` when no credentials |
| `agents/` | the eight agents, the program validator, the orchestrator |
| `evaluation/` | execution accuracy, abstention metrics, bootstrap CIs, exact McNemar, runners |
| `research/` | research agent team, hypothesis registry, approval gate, improvement loop |
| `api/app.py` | FastAPI endpoints for the frontend; `POST /api/ask/stream` sends one server-sent event per agent step, then the result, so the chat shows live progress |

## LLM use

`configs/full_llm.toml` enables Claude (`claude-opus-5-5`, via `anthropic` structured outputs) in three agents. Each LLM output is validated and falls back to the deterministic backend on any failure; the trace records which backend produced each step.

| Agent | What the LLM may decide | Guard |
| --- | --- | --- |
| Query analyst | operation, metric terms, periods | periods must appear in the question; operation must be in the enum |
| Reasoning agent | operation and evidence ids | only known ids, constants and `#n`; any written number rejects the program |
| Response agent | wording of one sentence | every number in it must already appear in the evidence or answer |

Refusals, truncation and API errors return `None`; the agent then uses its rules. Server-side model fallbacks are not configured, because the deterministic backend is the fallback.
