# The agents

Each agent has one responsibility, typed inputs and outputs, and only the tools it needs. Agents are classes in `src/finagents/agents/`; each returns only the state fields it owns.

| # | Agent | Responsibility | Reads | Writes | Tools |
| --- | --- | --- | --- | --- | --- |
| 1 | Orchestrator | Coordinate the workflow, route, enforce retry limits | all state | `route`, counters, `abstain_reason` | LangGraph, state manager |
| 2 | Query analyst | Understand the question: metric, periods, operation | question | `analysis` | rule parser, LLM structured output |
| 3 | Evidence researcher | Search tables and text; read values | question, analysis, round | `retrieved`, `found` | `search_evidence` (learned reranker), `extract_values` |
| 4 | Evidence auditor | Validate values, units, periods, relevance | `found`, `retrieved` | `audit` | relevance, period, unit and traceability checks |
| 5 | Reasoning agent | Construct the program over evidence ids | analysis, audit | `candidates` | program templates, LLM structured output, FinQA schema |
| – | Program validator | Reject malformed or ungrounded programs | candidate, audit | `validation` | program parser |
| 6 | Calculation agent | Execute deterministically | candidate, table | `calculation` | restricted executor |
| 7 | Verification agent | Independently verify operands, calculation, evidence | candidate, calculation, audit | `verification`, `attempts` | decimal recompute, evidence checker, plausibility rules |
| 8 | Response agent | Final answer and audit trail | everything | `final` | citation formatter, LLM structured output |

## Worked example (from the project plan; illustrative figures)

Question: *"What was the percentage growth in revenue from 2018 to 2019?"* against a table with revenue 1,000 (2018) and 1,200 (2019), in $ millions. This exact case is a test (`tests/test_agents.py::test_plan_example_answers_with_citations_trail_and_confidence`).

1. **Query analyst** → `{metric: "revenue", periods: ["2018", "2019"], operation: "percentage_change", required_evidence: 2}`.
2. **Evidence researcher** retrieves `table_1` (revenue row) and reads 1200 (2019, $ millions) and 1000 (2018, $ millions).
3. **Evidence auditor** checks both values describe revenue, cover both periods, have compatible units and re-occur in the source text; they become E1 and E2.
4. **Reasoning agent** writes `subtract(E1, E2), divide(#0, E2)` → `subtract(1200, 1000), divide(#0, 1000)`.
5. **Program validator** confirms the grammar and that both numbers are audited values.
6. **Calculation agent** executes: 200, then 0.2.
7. **Verification agent**: operands traced to `table_1`, exact decimal recomputation matches, the program uses subtract + divide as a percentage change needs, 0.2 is plausible, periods match, units consistent, no conflicting rows, no equally supported rival.
8. **Response agent**: **20.00%**, "Revenue changed by 20.00%, from 1000 (2018) to 1200 (2019).", formula `(1200 − 1000) / 1000 = 20.00%`, citation `table_1`, confidence 1.00.

## Verification checks

| Check | Kind | Fails when |
| --- | --- | --- |
| executes | hard | the executor returns an error |
| operands_grounded | hard | a number in the program is not found in its cited source text |
| recomputed | hard | exact decimal recomputation disagrees with the executor |
| matches_operation | hard | e.g. a percentage change without subtract and divide |
| plausible | hard | a share of a whole outside [0, 1], a percent change beyond ±1000%, a zero change, a yes/no mismatch |
| units_consistent | hard if `strict_units` | adding or subtracting a percentage and an amount |
| periods_match | soft | no operand from a named period (confidence × 0.6) |
| no_conflicting_evidence | soft; abstains if `abstain_on_conflict` | another row reports a different value for the same label, column and period |

After the checks, the abstention policy applies: confidence below `min_confidence` → research again or abstain; a different-valued candidate within `ambiguity_margin` that also passes the hard checks → ambiguous → research again or abstain.
