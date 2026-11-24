# Research log

Written by the Documentation Agent. Every entry is one iteration of the improvement loop, on train questions only.

## R20261124-163148 · iteration 1 · H1-confidence-up
- **Weakness:** wrong_answers
- **Outcome:** accepted; candidate saved to R20261124-163148-H1-confidence-up.toml and used as the new control
- **Hypothesis:** Raising the confidence threshold removes wrong answers faster than correct ones.
- **Acceptance:** wrong_answer_rate must decrease with exact McNemar p < 0.05 on the selection slice, move the same way on the confirmation slice, and execution_accuracy must not worsen by more than 1%
- **Change vs control (selection):** execution_accuracy -0.7%, coverage -4.7%, selective_accuracy +2.8%, wrong_answer_rate -4.0%, cites_all_gold +4.2%
  - leakage check: selection and confirmation are train slices; dev and test untouched
  - wrong_answer_rate improved: 25 vs 1, p = 8.05e-07
  - confirmation slice: wrong_answer_rate -4.3%
- **Candidate saved:** `configs/candidates/R20261124-163148-H1-confidence-up.toml`

## R20261124-163148 · iteration 2 · H5-stricter-relevance
- **Weakness:** wrong_answers
- **Outcome:** rejected
- **Hypothesis:** Auditing out weakly related values stops the reasoner binding the wrong row.
- **Acceptance:** wrong_answer_rate must decrease with exact McNemar p < 0.05 on the selection slice, move the same way on the confirmation slice, and execution_accuracy must not worsen by more than 1%
- **Change vs control (selection):** execution_accuracy +0.0%, coverage +0.3%, selective_accuracy -0.3%, wrong_answer_rate +0.3%, cites_all_gold -0.9%
  - leakage check: selection and confirmation are train slices; dev and test untouched
  - wrong_answer_rate: 1 better vs 3 worse, p = 0.625 (needs improvement with p < 0.05)
  - confirmation slice: wrong_answer_rate moved +0.0% (wrong direction)

## R20261124-163148 · iteration 3 · H6-wider-ambiguity-margin
- **Weakness:** wrong_answers
- **Outcome:** rejected
- **Hypothesis:** Abstaining when a rival answer is within 0.05 support avoids coin-flip answers.
- **Acceptance:** wrong_answer_rate must decrease with exact McNemar p < 0.05 on the selection slice, move the same way on the confirmation slice, and execution_accuracy must not worsen by more than 1%
- **Change vs control (selection):** execution_accuracy -1.0%, coverage -3.2%, selective_accuracy +0.9%, wrong_answer_rate -2.2%, cites_all_gold +1.3%
  - leakage check: selection and confirmation are train slices; dev and test untouched
  - wrong_answer_rate improved: 13 vs 0, p = 0.000244
  - confirmation slice: wrong_answer_rate -4.0%
  - guard execution_accuracy worsened by 1.0%

## R20261124-163148 · iteration 4 · H8-relaxed-units
- **Weakness:** no_program
- **Outcome:** rejected
- **Hypothesis:** Allowing percent/amount arithmetic recovers programs the unit check rejected.
- **Acceptance:** execution_accuracy must increase with exact McNemar p < 0.05 on the selection slice, move the same way on the confirmation slice, and wrong_answer_rate must not worsen by more than 1%
- **Change vs control (selection):** execution_accuracy +0.0%, coverage +0.3%, selective_accuracy -0.3%, wrong_answer_rate +0.3%, cites_all_gold -0.1%
  - leakage check: selection and confirmation are train slices; dev and test untouched
  - execution_accuracy: 0 better vs 0 worse, p = 1 (needs improvement with p < 0.05)
  - confirmation slice: execution_accuracy moved +0.0% (wrong direction)

## R20261124-164250 · iteration 1 · H1-confidence-up
- **Weakness:** wrong_answers
- **Outcome:** accepted; candidate saved to R20261124-164250-H1-confidence-up.toml and used as the new control
- **Hypothesis:** Raising the confidence threshold removes wrong answers faster than correct ones.
- **Acceptance:** wrong_answer_rate must decrease with exact McNemar p < 0.05 on the selection slice, move the same way on the confirmation slice, and execution_accuracy must not worsen by more than 1%
- **Change vs control (selection):** execution_accuracy -0.7%, coverage -4.7%, selective_accuracy +2.8%, wrong_answer_rate -4.0%, cites_all_gold +4.2%
  - leakage check: selection and confirmation are train slices; dev and test untouched
  - wrong_answer_rate improved: 25 vs 1, p = 8.05e-07
  - confirmation slice: wrong_answer_rate -4.3%
- **Candidate saved:** `configs/candidates/R20261124-164250-H1-confidence-up.toml`

## R20261124-164250 · iteration 2 · H5-stricter-relevance
- **Weakness:** wrong_answers
- **Outcome:** rejected
- **Hypothesis:** Auditing out weakly related values stops the reasoner binding the wrong row.
- **Acceptance:** wrong_answer_rate must decrease with exact McNemar p < 0.05 on the selection slice, move the same way on the confirmation slice, and execution_accuracy must not worsen by more than 1%
- **Change vs control (selection):** execution_accuracy +0.0%, coverage +0.3%, selective_accuracy -0.3%, wrong_answer_rate +0.3%, cites_all_gold -0.9%
  - leakage check: selection and confirmation are train slices; dev and test untouched
  - wrong_answer_rate: 1 better vs 3 worse, p = 0.625 (needs improvement with p < 0.05)
  - confirmation slice: wrong_answer_rate moved +0.0% (wrong direction)

## R20261124-164250 · iteration 3 · H6-wider-ambiguity-margin
- **Weakness:** wrong_answers
- **Outcome:** rejected
- **Hypothesis:** Abstaining when a rival answer is within 0.05 support avoids coin-flip answers.
- **Acceptance:** wrong_answer_rate must decrease with exact McNemar p < 0.05 on the selection slice, move the same way on the confirmation slice, and execution_accuracy must not worsen by more than 1%
- **Change vs control (selection):** execution_accuracy -1.0%, coverage -3.2%, selective_accuracy +0.9%, wrong_answer_rate -2.2%, cites_all_gold +1.3%
  - leakage check: selection and confirmation are train slices; dev and test untouched
  - wrong_answer_rate improved: 13 vs 0, p = 0.000244
  - confirmation slice: wrong_answer_rate -4.0%
  - guard execution_accuracy worsened by 1.0%

## R20261124-164250 · iteration 4 · H8-relaxed-units
- **Weakness:** no_program
- **Outcome:** rejected
- **Hypothesis:** Allowing percent/amount arithmetic recovers programs the unit check rejected.
- **Acceptance:** execution_accuracy must increase with exact McNemar p < 0.05 on the selection slice, move the same way on the confirmation slice, and wrong_answer_rate must not worsen by more than 1%
- **Change vs control (selection):** execution_accuracy +0.0%, coverage +0.3%, selective_accuracy -0.3%, wrong_answer_rate +0.3%, cites_all_gold -0.1%
  - leakage check: selection and confirmation are train slices; dev and test untouched
  - execution_accuracy: 0 better vs 0 worse, p = 1 (needs improvement with p < 0.05)
  - confirmation slice: execution_accuracy moved +0.0% (wrong direction)

## R20261124-164250 · iteration 5 · H9-llm-reasoning
- **Weakness:** weak_operation
- **Outcome:** awaiting human approval (89cab4c7): adds an external LLM dependency with per-question API cost
- **Hypothesis:** Letting Claude choose operations and evidence fixes question shapes the templates miss.

## R20261124-164250 · iteration 6 · H2-confidence-down
- **Weakness:** over_abstention
- **Outcome:** rejected
- **Hypothesis:** Lowering the confidence threshold recovers correct answers the agent withheld.
- **Acceptance:** execution_accuracy must increase with exact McNemar p < 0.05 on the selection slice, move the same way on the confirmation slice, and wrong_answer_rate must not worsen by more than 1%
- **Change vs control (selection):** execution_accuracy +0.7%, coverage +4.7%, selective_accuracy -2.8%, wrong_answer_rate +4.0%, cites_all_gold -4.2%
  - leakage check: selection and confirmation are train slices; dev and test untouched
  - confirmation slice: execution_accuracy +0.5%
  - execution_accuracy: 5 better vs 1 worse, p = 0.219 (needs improvement with p < 0.05)
  - guard wrong_answer_rate worsened by 4.0%

## R20261124-164250 · iteration 7 · H7-softer-period-penalty
- **Weakness:** over_abstention
- **Outcome:** rejected
- **Hypothesis:** A milder penalty for operands from unnamed periods keeps answers that were right.
- **Acceptance:** execution_accuracy must increase with exact McNemar p < 0.05 on the selection slice, move the same way on the confirmation slice, and wrong_answer_rate must not worsen by more than 1%
- **Change vs control (selection):** execution_accuracy +0.0%, coverage +0.0%, selective_accuracy +0.0%, wrong_answer_rate +0.0%, cites_all_gold +0.0%
  - leakage check: selection and confirmation are train slices; dev and test untouched
  - execution_accuracy: 0 better vs 0 worse, p = 1 (needs improvement with p < 0.05)
  - confirmation slice: execution_accuracy moved +0.0% (wrong direction)
