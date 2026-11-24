# FinAgents contributor guidance

Multi-agent financial question answering over FinQA. Read `README.md` and `docs/architecture.md` first.

## Rules

- **Numbers come from tools, never from a model.** Agents reference evidence by id; values are substituted from audited evidence; all arithmetic goes through `core/executor.py`. Do not add a code path where an LLM output becomes a number.
- **Agents never read answers.** Agent inputs come from `core.records.inference_input`. Only `evaluation/` and the reranker's training (train split) may call `gold()`.
- **Train for choices, dev for reporting, test never.** `load_split` refuses `test`. Research-loop selection uses train slices only.
- **The research loop is bounded.** New experiments are new entries in `research/registry.py` touching only `TUNABLE` settings. Protected actions go through `research/approvals.py`.
- Use `uv` for Python and `pnpm` in `frontend/`. Keep secrets in environment variables (`ANTHROPIC_API_KEY`); never commit them.
- Do not invent results. Every number in `README.md` and `docs/` comes from a file in `runs/` or the research log.

## Checks

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
cd frontend && pnpm lint && pnpm build
```
