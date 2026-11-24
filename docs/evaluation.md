# Evaluation

## Protocol

- **Data.** FinQA train (6,251) and dev (883), pinned files with SHA-256 checks ([background/dataset-card.md](background/dataset-card.md)). The official test split is not loaded by any code path; `load_split("test")` raises.
- **Fitting and selection on train only.** The reranker and TF-IDF are fit on 90% of train, the reranker model is selected on the remaining 10%; agent rules were written from train questions; the research loop selects on train slices 1000–1599 and confirms on 2000–2599.
- **Dev for reporting.** Each system configuration ran once on dev. No setting was changed after looking at dev results.
- **Metric.** Official FinQA execution accuracy: an answer is correct when its value equals `exe_ans` after five-decimal rounding; every question stays in the denominator, so abstaining counts as not correct. The executor reproduces `exe_ans` for all 7,134 train + dev gold programs.
- **Abstention metrics.** Coverage (share answered), accuracy when answering (selective accuracy), wrong-answer rate (share of all questions answered wrongly), cites all gold (answered questions whose citations include every gold evidence segment).
- **Uncertainty.** Percentile bootstrap 95% intervals (2,000 resamples, seed 42); exact two-sided McNemar tests on paired outcomes.

## Reproduce

```bash
uv run finagents train
uv run finagents retrieval --split dev
uv run finagents evaluate --split dev --configs single_pass no_abstention full full_bm25 candidates/R20261124-163148-H1-confidence-up
```

Outputs: `runs/<config>_dev.jsonl` (every answer, citation, calculation and trace), `runs/<config>_dev_summary.json`, `runs/comparison_dev.json`, `runs/retrieval_dev.json`.

## Results (dev, 883 questions, run 2026-11-24)

| System | Exec. accuracy [95% CI] | Coverage | Selective acc. | Wrong-answer rate | Cites all gold | ms/q |
| --- | --- | --- | --- | --- | --- | --- |
| Single pass (ablation) | 25.9% [23.1, 28.8] | 74.8% | 34.7% | 48.8% | 54.7% | 2.9 |
| No abstention (ablation) | 26.8% [23.9, 29.8] | 75.2% | 35.7% | 48.4% | 53.9% | 4.2 |
| **Full system** | 18.8% [16.3, 21.4] | 47.1% | 39.9% | 28.3% | 59.1% | 6.9 |
| Full system, BM25 | 17.8% [15.3, 20.4] | 45.9% | 38.8% | 28.1% | 55.8% | 4.9 |
| Research candidate H1 | 18.4% [16.0, 20.9] | 43.1% | 42.5% | 24.8% | 61.4% | 7.2 |

Paired against the single pass:

| System | Wrong answers avoided / added | p | Correct gained / lost | p |
| --- | --- | --- | --- | --- |
| No abstention | 10 / 6 | 0.45 | 10 / 2 | 0.039 |
| Full system | 182 / 1 | 3e-53 | 2 / 65 | 3e-17 |
| Full, BM25 | 194 / 11 | 2e-44 | 3 / 75 | 5e-19 |
| Candidate H1 | 213 / 1 | 2e-62 | 2 / 69 | 2e-18 |

Why the full system abstained: 96 unsupported question shapes, 127 no calculation fits the evidence, 130 ambiguous (an equally supported rival answer), 108 low confidence, 6 failed checks. 66 of the withheld best candidates were in fact correct.

**Retrieval** (all gold segments in the top k, ranking with the question alone):

| Retriever | Top 3 | Top 5 | Top 10 |
| --- | --- | --- | --- |
| BM25 | 55.5% | 70.9% | 87.9% |
| TF-IDF (train IDF) | 45.4% | 60.9% | 82.7% |
| **Learned reranker (MLP)** | **72.5%** | **82.7%** | **92.2%** |

## Claude-assisted system: 100-question dev sample (run 2026-11-24)

`uv run finagents evaluate --split dev --configs full full_llm --sample 100 --seed 42 --workers 6`. A seeded random sample of 100 dev questions; both systems answered the same questions. Claude (`claude-opus-5-5`) ran the query analyst, reasoning and response agents; the thresholds are the offline system's (chosen on train for the template reasoner). Cost: 335 API calls, 343,411 input and 53,461 output tokens, about $2.44.

| System | Exec. accuracy [95% CI] | Coverage | Accuracy when answering | Wrong-answer rate | Cites all gold |
| --- | --- | --- | --- | --- | --- |
| Full, offline | 21% [13, 29] | 54% | 38.9% | 33% | 61.1% |
| **Full with Claude** | 19% [12, 27] | 25% | **76.0%** | **6%** | 64.0% |

Paired: Claude avoided 29 of the offline system's wrong answers and added 2 (exact McNemar p = 5e-7); correct answers 14 gained, 16 lost (p = 0.86, no detectable difference).

**What limits it.** 55 of its 75 abstentions are "low confidence", and 47 of all its withheld best candidates were correct. The confidence score is the template reasoner's label-match support, which undervalues Claude's choices; the 0.60 threshold was tuned for templates. This is an observation on a dev sample, so the threshold is **not** changed here: a confidence rule for the Claude system has to be chosen on train first, then checked on dev. With 100 questions the intervals are wide.

## Interpretation

- **The agentic controls buy reliability, not raw accuracy.** Iterative research plus verification alone (no-abstention ablation) adds 0.9 points of accuracy (10 gained, 2 lost, p = 0.039). Abstention then removes most wrong answers: 182 avoided against 1 added. For an analyst, a withheld answer is cheaper than a confident wrong number.
- **Retrieval matters.** The learned reranker raises all-gold recall@5 by 11.8 points over BM25, and the full system with it cites all gold evidence on 59.1% of answers vs 55.8% with BM25.
- **The offline reasoner is the ceiling.** Template slot filling covers seven calculation shapes; 96 questions (10.9%) are outside them, and multi-step questions are not planned. With Claude, answers given are far more often right (76% vs 39% on the 100-question sample), but the confidence gate needs its own train-chosen threshold before coverage can be judged.

## Limitations

- One deterministic run per system; no seeds to vary in the offline agents.
- Dev results only; no test-set number exists.
- The confidence score is a heuristic support score, not a calibrated probability.
- Some FinQA labels are inconsistent (e.g. dev #3: answer text 18.6, `exe_ans` 19.2); scoring follows the official `exe_ans`.
