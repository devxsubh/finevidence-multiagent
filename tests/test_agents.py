"""The agents and the orchestrated graph, on synthetic reports and a fake LLM."""

import pytest

from finagents.agents import Orchestrator
from finagents.agents.query_analyst import analyze_with_rules
from finagents.agents.response import display_value
from finagents.config import SystemConfig
from finagents.retrieval import Bm25Retriever
from finagents.state import Operation, QueryAnalysis
from fixtures import record


class FakeLLM:
    """Returns canned structured outputs by schema name; records what it was asked."""

    name = "fake"

    def __init__(self, outputs):
        self.outputs = outputs
        self.last_error = None
        self.calls = []

    def structured(self, *, system, prompt, schema):
        self.calls.append(schema.__name__)
        value = self.outputs.get(schema.__name__)
        if value is None:
            self.last_error = "no output"
            return None
        return schema(**value)


def run(question=None, config=None, llm=None, **changes):
    system = Orchestrator(
        config or SystemConfig(retriever="bm25"), llm=llm, retriever=Bm25Retriever()
    )
    return system.run(record(question, **changes))


def agents_in(result):
    return [step["agent"] for step in result["trace"]]


def test_plan_example_answers_with_citations_trail_and_confidence():
    result = run()
    answer = result["answer"]
    assert (
        answer["status"] == "answered" and answer["value"] == 0.2 and answer["display"] == "20.00%"
    )
    assert answer["program"] == "subtract(1200, 1000), divide(#0, 1000)"
    assert [c["segment_key"] for c in answer["citations"]] == ["table_1"]
    assert answer["formula"] == "(1200 − 1000) / 1000 = 20.00%"
    assert answer["calculation"][0].startswith("Step 1: subtract(1200, 1000) = 200")
    assert answer["confidence"] >= 0.6
    assert {c["name"] for c in answer["checks"]} >= {
        "operands_grounded",
        "recomputed",
        "matches_operation",
    }
    assert agents_in(result) == [
        "orchestrator",
        "query_analyst",
        "evidence_researcher",
        "evidence_auditor",
        "reasoning",
        "program_validator",
        "calculation",
        "verification",
        "response",
    ]


def test_query_analyst_rules():
    analysis = analyze_with_rules("what was the percentage growth in revenue from 2018 to 2019?")
    assert analysis.operation is Operation.PERCENTAGE_CHANGE
    assert analysis.periods == ["2018", "2019"] and analysis.entity_terms == ["revenue"]
    assert analyze_with_rules(
        "what is the ratio of revenue to total assets in 2019?"
    ).whole_terms == ["assets"]


def test_unsupported_question_abstains_without_research():
    result = run("describe the company's strategy")
    assert result["answer"]["status"] == "abstained"
    assert "unsupported" in result["answer"]["abstention_reason"]
    assert "evidence_researcher" not in agents_in(result)


def test_missing_evidence_triggers_more_research_then_abstains():
    result = run("what was the change in dividends paid from 2016 to 2017?")
    assert result["answer"]["status"] == "abstained"
    assert agents_in(result).count("evidence_researcher") == 3  # the research-round budget
    assert result["research_rounds"] == 3


def test_equally_supported_rival_answers_abstain_as_ambiguous():
    # "gross margin" matches a percent row and a dollar row equally: two different answers.
    table = [
        ["( in millions )", "2019", "2018"],
        ["gross margin %", "27.3%", "27.1%"],
        ["gross margin", "$ 2259", "$ 1708"],
    ]
    answer = run("what was the change in gross margin from 2018 to 2019?", table=table)["answer"]
    assert answer["status"] == "abstained" and answer["abstention_reason"].startswith("ambiguous")


def test_orchestrator_routing_policy():
    from finagents.state import Verification

    system = Orchestrator(
        SystemConfig(retriever="bm25", max_research_rounds=2), retriever=Bm25Retriever()
    )
    base = {"candidates": [object(), object()], "candidate_index": 0, "research_round": 1}
    failed = Verification(verified=False, confidence=0.9, checks=[], failure="checks", reason="x")
    assert system.decide("verification", {**base, "verification": failed}) == (
        "program_validator",
        {"candidate_index": 1},
    )
    last = {**base, "candidate_index": 1, "verification": failed}
    assert system.decide("verification", last) == ("evidence_researcher", {"research_round": 2})
    spent = {**last, "research_round": 2}
    assert system.decide("verification", spent)[0] == "abstain"
    conflict = Verification(
        verified=False, confidence=0.9, checks=[], failure="conflict", reason="c"
    )
    assert system.decide("verification", {**base, "verification": conflict})[0] == "abstain"


def test_single_pass_ablation_never_retries():
    config = SystemConfig(retriever="bm25", max_research_rounds=1, verify=False, abstain=False)
    result = run("what was the change in dividends paid from 2016 to 2017?", config=config)
    assert agents_in(result).count("evidence_researcher") == 1


def test_display_value():
    analysis = analyze_with_rules("what was the percentage change in revenue from 2018 to 2019?")
    assert display_value(analysis, analysis.metric, 0.2) == "20.00%"
    assert display_value(analyze_with_rules("what was the change in revenue"), "x", 200.0) == "200"


# LLM paths -----------------------------------------------------------------------------------

LLM_CONFIG = SystemConfig(retriever="bm25", use_llm=True)


def test_llm_query_analysis_is_validated_against_the_question():
    llm = FakeLLM(
        {
            "QueryAnalysis": {
                "metric": "revenue",
                "entity_terms": ["Revenue"],
                "periods": ["2018", "2019", "2030"],
                "operation": "percentage_change",
                "required_evidence": 2,
            }
        }
    )
    result = run(config=LLM_CONFIG, llm=llm)
    assert result["analysis"]["periods"] == ["2018", "2019"]  # 2030 is not in the question
    assert result["analysis"]["backend"] == "llm:fake"
    assert result["answer"]["value"] == 0.2


def test_llm_program_over_evidence_ids_is_used_and_numbers_come_from_evidence():
    llm = FakeLLM(
        {"ProgramPlan": {"template": "subtract(E1, E2), divide(#0, E2)", "rationale": "growth"}}
    )
    result = run(config=LLM_CONFIG, llm=llm)
    reasoning = next(s for s in result["trace"] if s["agent"] == "reasoning")
    assert reasoning["details"]["backend"] == "llm:fake"
    assert result["answer"]["status"] == "answered"


@pytest.mark.parametrize(
    "template", ["subtract(1200, 1000)", "subtract(E99, E1)", "import(E1, E2)"]
)
def test_llm_program_that_writes_numbers_or_bad_ids_is_rejected(template):
    llm = FakeLLM({"ProgramPlan": {"template": template, "rationale": "x"}})
    result = run(config=LLM_CONFIG, llm=llm)
    reasoning = next(s for s in result["trace"] if s["agent"] == "reasoning")
    assert "rejected" in reasoning["details"]["backend"]
    assert result["answer"]["value"] == 0.2  # the template candidates still answer


def test_llm_sentence_with_invented_numbers_falls_back_to_template():
    llm = FakeLLM({"Sentence": {"sentence": "Revenue grew 35% to $9,999 million."}})
    result = run(config=LLM_CONFIG, llm=llm)
    response = next(s for s in result["trace"] if s["agent"] == "response")
    assert "numbers not in the evidence" in response["details"]["backend"]
    assert "9,999" not in result["answer"]["explanation"]


def test_llm_unavailable_falls_back_to_rules():
    result = run(config=LLM_CONFIG, llm=FakeLLM({}))
    assert result["analysis"]["backend"] == "rules" and result["answer"]["value"] == 0.2


def test_query_analysis_schema_rejects_unknown_operation():
    with pytest.raises(ValueError):
        QueryAnalysis(
            metric="x", entity_terms=["x"], periods=[], operation="guess", required_evidence=1
        )
