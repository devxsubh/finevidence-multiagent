"""Small synthetic FinQA-shaped reports shared by the tests. Not real report data."""

from __future__ import annotations

import copy

REVENUE_REPORT = {
    "id": "SYNTH/2019/page_1.pdf-1",
    "pre_text": [
        "the company reports results in two segments .",
        "revenue increased due to higher volume in 2019 .",
    ],
    "post_text": ["interest expense was $ 40 million in 2019 ."],
    "table": [
        ["( in millions )", "2019", "2018"],
        ["revenue", "$ 1200", "$ 1000"],
        ["operating expenses", "$ 900", "$ 800"],
        ["total assets", "$ 5000", "$ 4500"],
    ],
    "qa": {
        "question": "what was the percentage growth in revenue from 2018 to 2019?",
        "exe_ans": 0.2,
        "answer": "20%",
        "program": "subtract(1200, 1000), divide(#0, 1000)",
        "gold_inds": {"table_1": "revenue of 2019 is $ 1200 ; of 2018 is $ 1000 ;"},
    },
}


def record(question: str | None = None, **changes) -> dict:
    """A deep copy of the revenue report, optionally with another question or fields."""
    item = copy.deepcopy(REVENUE_REPORT)
    if question:
        item["qa"]["question"] = question
    item.update(changes)
    return item
