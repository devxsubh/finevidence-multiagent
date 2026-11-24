"""Deterministic core: records allowlist, value parsing, program parser, executor."""

from decimal import Decimal

import pytest

from finagents.core.executor import execute_program, to_number
from finagents.core.program import parse_program
from finagents.core.records import assert_inference_safe, gold, inference_input
from finagents.core.segments import build_segments, extract_mentions
from finagents.core.values import parse_value, unit_of, units_compatible
from fixtures import record


def test_inference_input_drops_answers_and_gold_evidence():
    payload = inference_input(record())
    assert set(payload) == {"id", "question", "table", "pre_text", "post_text"}
    assert "qa" not in payload and "1200" in payload["table"][1][1]
    with pytest.raises(ValueError):
        assert_inference_safe({**payload, "answer": "20%"})


def test_gold_is_separate_from_inference_input():
    assert gold(record())["exe_ans"] == 0.2


@pytest.mark.parametrize(
    ("raw", "value", "percent"),
    [
        ("$ 1,234.5", Decimal("1234.5"), False),
        ("( 12 )", Decimal("-12"), False),
        ("4.3%", Decimal("4.3"), True),
        ("$ -4063 ( 4063 )", Decimal("-4063"), False),
        (".5", Decimal(".5"), False),
    ],
)
def test_parse_value(raw, value, percent):
    parsed = parse_value(raw)
    assert parsed is not None and parsed.value == value and parsed.is_percent is percent


@pytest.mark.parametrize("raw", ["2019", "n/a", "december 31 2019", "revenue", ""])
def test_parse_value_rejects_years_and_text(raw):
    assert parse_value(raw) is None


def test_units():
    assert unit_of("$ 1200", "( in millions )") == "$ millions"
    assert unit_of("4.3%", is_percent=True) == "%"
    assert not units_compatible("%", "$ millions")
    assert units_compatible("$ millions", "millions")


def test_parser_accepts_finqa_programs_and_rejects_code():
    assert parse_program("subtract(1200, 1000), divide(#0, 1000)").is_valid
    assert parse_program("table_max(revenue, none)").is_valid
    for bad in ["__import__('os')", "divide(#1, 2)", "pow(2, 3)", "add(1)", "add(1, 2"]:
        assert not parse_program(bad).is_valid, bad


def test_executor_follows_official_rules():
    table = record()["table"]
    assert execute_program("subtract(1200, 1000), divide(#0, 1000)", table).value == 0.2
    assert execute_program("greater(1200, 1000)", table).value == "yes"
    assert execute_program("table_max(revenue, none)", table).value == 1200
    assert execute_program("divide(1, 3)", table).value == 0.33333
    assert to_number("const_m1") == -1.0 and to_number("12%") == 0.12


def test_executor_reports_errors_instead_of_raising():
    table = record()["table"]
    assert "arithmetic error" in execute_program("divide(5, 0)", table).error
    assert "no numeric table row" in execute_program("table_sum(missing row, none)", table).error


def test_segments_use_finqa_keys_and_mentions_carry_period_and_unit():
    segments = build_segments(inference_input(record()))
    assert [s.key for s in segments][:2] == ["table_0", "table_1"] and segments[-1].key == "text_2"
    revenue = [m for m in extract_mentions(segments) if m.label == "revenue"]
    assert {(m.literal, m.periods, m.unit) for m in revenue} == {
        ("1200", ("2019",), "$ millions"),
        ("1000", ("2018",), "$ millions"),
    }
