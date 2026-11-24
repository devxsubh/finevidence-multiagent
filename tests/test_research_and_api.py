"""Research loop safeguards (approval gate, critic, registry) and the HTTP API."""

import pytest
from fastapi.testclient import TestClient

from finagents.research import approvals
from finagents.research.registry import HYPOTHESES, apply_change, by_id
from finagents.research.team import DocumentationAgent, ExperimentDesigner, ResearchCritic
from fixtures import record


def test_apply_change_is_bounded_to_tunable_settings():
    base = {"min_confidence": 0.6, "initial_k": 5, "strict_units": True}
    assert apply_change(base, {"min_confidence": "+0.1"})["min_confidence"] == 0.7
    assert apply_change(base, {"initial_k": "+3"})["initial_k"] == 8
    with pytest.raises(ValueError):
        apply_change(base, {"split": "test"})


def test_every_hypothesis_uses_only_tunable_settings_and_protected_ones_explain_why():
    for h in HYPOTHESES:
        apply_change(
            {
                "min_confidence": 0.6,
                "initial_k": 5,
                "max_research_rounds": 3,
                "min_relevance": 0.25,
                "ambiguity_margin": 0.01,
                "period_mismatch_penalty": 0.6,
                "strict_units": True,
                "use_llm": False,
                "llm_agents": [],
            },
            h.change,
        )
        assert not h.protected or h.reason_protected


def test_approval_requests_are_filed_once_and_decided_by_a_human(tmp_path):
    path = tmp_path / "approvals.json"
    first = approvals.request("run_hypothesis", "H9-llm-reasoning", "cost", path)
    again = approvals.request("run_hypothesis", "H9-llm-reasoning", "cost", path)
    assert (
        first["id"] == again["id"]
        and approvals.status("run_hypothesis", "H9-llm-reasoning", path) == "pending"
    )
    approvals.decide(first["id"], "approved", path)
    assert approvals.status("run_hypothesis", "H9-llm-reasoning", path) == "approved"


def _rows(correct, answered):
    return [{"correct": c, "answered": a} for c, a in zip(correct, answered, strict=True)]


def test_critic_rejects_an_insignificant_change_and_accepts_a_clear_one():
    h = by_id("H2-confidence-down")
    experiment = ExperimentDesigner().design(h, {"name": "full"})
    control = _rows([False] * 40, [True] * 40)
    summary = {"execution_accuracy": 0.0, "wrong_answer_rate": 1.0}
    same = ResearchCritic().review(
        h, experiment, control, control, summary, summary, summary, summary
    )
    assert not same["accepted"]
    better = _rows([True] * 30 + [False] * 10, [True] * 40)
    better_summary = {"execution_accuracy": 0.75, "wrong_answer_rate": 0.25}
    verdict = ResearchCritic().review(
        h, experiment, control, better, summary, better_summary, summary, better_summary
    )
    assert verdict["accepted"], verdict["problems"]


def test_documentation_agent_writes_the_log(tmp_path, monkeypatch):
    import finagents.research.team as team

    monkeypatch.setattr(team, "RESEARCH_LOG", tmp_path / "research-log.md")
    monkeypatch.setattr(team, "PROJECT_ROOT", tmp_path)
    DocumentationAgent().record("R1", {"iteration": 1, "hypothesis": "H1", "outcome": "rejected"})
    assert "R1 · iteration 1" in (tmp_path / "research-log.md").read_text()
    assert (tmp_path / "runs" / "research" / "R1.json").exists()


# API ----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    from finagents.api.app import app

    return TestClient(app)


def test_health_and_agent_roster(client):
    assert client.get("/api/health").json()["status"] == "ok"
    names = [a["name"] for a in client.get("/api/agents").json()]
    assert names[0] == "orchestrator" and "verification" in names and len(names) == 9


def test_ask_with_a_user_report(client):
    item = record()
    body = {
        "config": "full_bm25",
        "record": {
            "question": item["qa"]["question"],
            "table": item["table"],
            "pre_text": item["pre_text"],
            "post_text": item["post_text"],
        },
    }
    response = client.post("/api/ask", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"]["value"] == 0.2 and "reference" not in data
    assert data["trace"][0]["agent"] == "orchestrator"


def test_ask_rejects_bad_requests(client):
    assert (
        client.post("/api/ask", json={"config": "nope", "split": "dev", "index": 0}).status_code
        == 400
    )
    assert client.post("/api/ask", json={"config": "full_bm25"}).status_code == 400
    empty = {"config": "full_bm25", "record": {"question": "what is revenue?"}}
    assert client.post("/api/ask", json=empty).status_code == 400


def test_streaming_sends_each_agent_step_then_the_result(client):
    item = record()
    body = {
        "config": "full_bm25",
        "record": {
            "question": item["qa"]["question"],
            "table": item["table"],
            "pre_text": item["pre_text"],
            "post_text": item["post_text"],
        },
    }
    with client.stream("POST", "/api/ask/stream", json=body) as response:
        text = "".join(response.iter_text())
    events = [line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("event: ")]
    assert events[0] == "step" and events[-1] == "result" and events.count("step") == 9


def test_a_new_question_about_a_dataset_report_has_no_reference(client):
    data = client.post(
        "/api/ask",
        json={
            "config": "full_bm25",
            "split": "dev",
            "index": 3,
            "question": "what was net revenue in 2009?",
        },
    ).json()
    assert data["question"] == "what was net revenue in 2009?" and "reference" not in data
