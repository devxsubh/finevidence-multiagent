# FinAgents: a multi-agent assistant for financial reports

FinAgents is a chat assistant for questions about company financial reports ("What was the percentage change in net revenue from 2009 to 2010?"). Behind each reply, eight cooperating agents plan the question, find the figures, audit them, write a calculation, run it on a deterministic calculator and verify the result. Every answer shows the evidence it used, the exact calculation, the checks it passed and a confidence. When the evidence isn't good enough, the assistant says so and explains why instead of guessing.

**Core rule: language models decide what to do; deterministic tools decide what the numbers are.** Agents refer to figures only by evidence id (E1, E2). Values are substituted from audited evidence, and every calculation runs on a restricted executor that reproduces the official FinQA evaluator on all 7,134 train and dev gold programs.

It is built and evaluated on [FinQA](https://github.com/czyssrs/FinQA) (8,281 expert-written questions over S&P 500 earnings reports). A second team of seven research agents experiments on the system itself, under human approval gates.

## Run it

```bash
uv sync --all-extras                      # Python 3.11–3.13
uv run finagents train                    # fit TF-IDF + the learned reranker on train (~20 s, CPU)
uv run finagents serve                    # API on http://localhost:8000 (docs at /docs)

cd frontend && pnpm install && pnpm dev   # app on http://localhost:3000
```

`data/raw/train.json` and `dev.json` are the pinned FinQA files (SHA-256 in [data/README.md](data/README.md)); they are Git-ignored. Claude is optional: set `ANTHROPIC_API_KEY` and choose the `full_llm` system to let it analyse questions, choose evidence and phrase answers. Without a key, every agent uses its deterministic backend.

## The app

| Page | What it does |
| --- | --- |
| **Assistant** | Open a FinQA report (or paste your own table and text) and chat about it. Each reply streams the agents' progress live, then shows the answer, evidence ids, formula, checks and the full agent trace. The report sits beside the chat as a workpaper, with the selected answer's figures highlighted and ticked. |
| **Evaluation** | Dev results for the full system and its ablations, with confidence intervals, paired significance tests, abstention reasons and retrieval recall. |
| **Research lab** | Start a research session, watch the improvement loop, read every iteration's outcome and approve or reject protected actions. |

The command line does the same: `uv run finagents ask --index 3` answers dev question 3 with its audit trail.

## How it works

```
Financial question
      │
      ▼
 1 Orchestrator (LangGraph): plans, routes, enforces retry limits, owns all state
      │
      ├─▶ 2 Query analyst        metric, periods, operation             (rules or Claude, validated)
      ├─▶ 3 Evidence researcher  search tables + text, read values      (learned reranker)
      ├─▶ 4 Evidence auditor     relevance, periods, units, traceability ─ insufficient ─▶ research again / abstain
      ├─▶ 5 Reasoning agent      program over evidence ids              (templates or Claude)
      ├─▶   Program validator    grammar + grounding gate              ─ invalid ─▶ next candidate
      ├─▶ 6 Calculation agent    deterministic executor (no eval)
      ├─▶ 7 Verification agent   decimal recompute, operands, plausibility, conflicts, rival answers
      │                           ─ failed ─▶ next candidate │ low confidence ─▶ research again │ conflict ─▶ abstain
      ▼
 8 Response agent  answer + citations + calculation trail + confidence, or a reasoned abstention
```

Details: [docs/architecture.md](docs/architecture.md) (graph, state, routing, LLM guards) and [docs/agents.md](docs/agents.md) (each agent's inputs, outputs and tools, with a worked example).

## Results on FinQA dev (883 questions)

Official execution accuracy (an abstention counts as not correct). Settings were chosen on train; each system ran once on dev; the test split is never used. All systems run offline (no LLM). Bootstrap 95% CI in brackets.

| System | Exec. accuracy | Coverage | Accuracy when answering | Wrong-answer rate | Cites all gold |
| --- | --- | --- | --- | --- | --- |
| Single pass (ablation) | 25.9% [23.1, 28.8] | 74.8% | 34.7% | 48.8% | 54.7% |
| No abstention (ablation) | 26.8% [23.9, 29.8] | 75.2% | 35.7% | 48.4% | 53.9% |
| **Full multi-agent system** | 18.8% [16.3, 21.4] | 47.1% | **39.9%** | **28.3%** | **59.1%** |
| Full system, BM25 retrieval | 17.8% [15.3, 20.4] | 45.9% | 38.8% | 28.1% | 55.8% |
| Research-loop candidate (H1) | 18.4% [16.0, 20.9] | 43.1% | **42.5%** | **24.8%** | 61.4% |

Against the single pass, the full system avoids 182 wrong answers and adds 1 (exact McNemar p ≈ 3e-53), at the cost of 65 correct answers withheld. It trades coverage for reliability: when it answers it is right more often, and wrong answers fall from 49% to 28% of all questions. The offline template reasoner caps absolute accuracy. With Claude choosing operations and evidence (`configs/full_llm.toml`), a 100-question dev sample gave 76% accuracy when answering and a 6% wrong-answer rate, but only 25% coverage: its confidence gate, tuned for the templates, withheld 47 answers that were correct. Re-tuning that gate on train is the next step ([docs/evaluation.md](docs/evaluation.md)).

Evidence retrieval, all gold facts in the top 5: learned reranker **82.7%**, BM25 70.9%, TF-IDF 60.9%. Full protocol and analysis: [docs/evaluation.md](docs/evaluation.md).

## Research agents

```bash
uv run finagents research run --iterations 5   # train questions only; writes docs/research-log.md
uv run finagents research approvals            # protected actions waiting for a human
uv run finagents research approve <id>
```

Seven agents (research analyst, dataset agent, experiment designer, ML engineer, evaluation agent, research critic, documentation agent) diagnose weaknesses, test one bounded change per iteration on train, and keep it only if the critic finds a significant improvement (exact McNemar p < 0.05) that also holds on a second train slice. Using an external LLM, evaluating on test or promoting a candidate to the default all wait for human approval. See [docs/research-loop.md](docs/research-loop.md) and the agents' own log, [docs/research-log.md](docs/research-log.md).

## Project map

```text
finevidence-multiagent/
├── src/finagents/
│   ├── core/          deterministic core: records allowlist, value parsing, program parser, executor
│   ├── retrieval/     BM25, TF-IDF and the learned (neural) reranker
│   ├── llm/           model-agnostic LLM interface; Claude backend
│   ├── agents/        the eight agents and the LangGraph orchestrator
│   ├── evaluation/    FinQA metrics, abstention metrics, significance tests, runners
│   ├── research/      research agent team, hypothesis registry, approval gate, improvement loop
│   ├── api/           FastAPI app used by the frontend (including the streaming chat endpoint)
│   ├── state.py       typed state and the structured messages agents exchange
│   ├── config.py      system configuration
│   └── cli.py         `finagents` command
├── frontend/          Next.js app: Assistant (chat), Evaluation, Research lab
├── configs/           one TOML per system variant; candidates/ from the research loop
├── tests/             unit and integration tests (pytest)
├── docs/              architecture, agents, evaluation, research loop and log, background
├── data/raw/          FinQA train/dev (Git-ignored)
├── artifacts/         trained retriever weights (Git-ignored; `finagents train`)
└── runs/              evaluation outputs and research sessions (Git-ignored)
```

## Checks

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
cd frontend && pnpm lint && pnpm build
```

FinAgents is a research prototype. It does not provide financial or investment advice.
