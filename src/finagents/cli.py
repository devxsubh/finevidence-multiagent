"""Command line: ``uv run finagents <command>``.

train                       fit TF-IDF and the learned reranker on train (about 20 s)
ask --split dev --index 3   answer one question and print the audit trail
ask --record report.json    your own report: {question, table, pre_text, post_text}
evaluate                    full system vs ablations on dev, with paired tests
retrieval                   all-gold recall@k of BM25, TF-IDF and the learned reranker
research run                run the autonomous improvement loop on train
research approvals          list protected actions waiting for a human
research approve <id>       approve (or `research reject <id>`)
serve                       start the API for the frontend on http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finagents.core.records import gold, load_split


def _render(result: dict) -> str:
    a = result["answer"]
    lines = [f"## {result['question']}", ""]
    if a["status"] == "answered":
        lines += [
            f"**Answer: {a['display']}**  (confidence {a['confidence']:.2f})",
            "",
            a["explanation"],
            "",
        ]
        if a["formula"]:
            lines += [f"Formula: `{a['formula']}`", ""]
    else:
        lines += [f"**Abstained:** {a['abstention_reason']}", ""]
    if a["citations"]:
        lines.append("Evidence:")
        lines += [f"- [{c['segment_key']}] {c['text']}" for c in a["citations"]]
        lines.append("")
    if a["calculation"]:
        lines += ["Calculation:"] + [f"- {line}" for line in a["calculation"]] + [""]
    lines.append("Agent trace:")
    for step in result["trace"]:
        route = f"  → {step['route']}" if step.get("route") else ""
        lines.append(f"- {step['agent']}: {step['summary']}{route}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="finagents", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train")
    ask = sub.add_parser("ask")
    ask.add_argument("--split", default="dev")
    ask.add_argument("--index", type=int, default=0)
    ask.add_argument("--record", type=Path)
    ask.add_argument("--config", default="full")
    ask.add_argument("--json", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--split", default="dev")
    ev.add_argument(
        "--configs", nargs="+", default=["single_pass", "no_abstention", "full", "full_bm25"]
    )
    ev.add_argument("--limit", type=int, help="first N questions")
    ev.add_argument("--sample", type=int, help="seeded random sample of N questions")
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--workers", type=int, default=1, help="questions answered in parallel")
    rt = sub.add_parser("retrieval")
    rt.add_argument("--split", default="dev")
    rs = sub.add_parser("research")
    rs_sub = rs.add_subparsers(dest="action", required=True)
    run = rs_sub.add_parser("run")
    run.add_argument("--config", default="full")
    run.add_argument("--iterations", type=int, default=3)
    rs_sub.add_parser("approvals")
    for verb in ("approve", "reject"):
        rs_sub.add_parser(verb).add_argument("request_id")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.command == "train":
        from finagents.retrieval import ARTIFACTS
        from finagents.retrieval.learned import train_reranker

        print("training on train split only")
        report = train_reranker(load_split("train"), ARTIFACTS)
        print(json.dumps(report, indent=2))
    elif args.command == "ask":
        from finagents.agents import Orchestrator
        from finagents.config import load_config

        record = (
            json.loads(args.record.read_text())
            if args.record
            else load_split(args.split)[args.index]
        )
        result = Orchestrator(load_config(args.config)).run(record)
        if not args.record:
            result["reference"] = gold(record)[
                "exe_ans"
            ]  # shown after the run; never given to agents
        print(json.dumps(result, indent=2) if args.json else _render(result))
        if not args.record and not args.json:
            print(f"\n(reference exe_ans: {result['reference']})")
    elif args.command == "evaluate":
        from finagents.evaluation.runner import compare

        comparison = compare(
            args.configs,
            args.split,
            limit=args.limit,
            sample=args.sample,
            seed=args.seed,
            workers=args.workers,
        )
        keys = (
            "execution_accuracy",
            "coverage",
            "selective_accuracy",
            "wrong_answer_rate",
            "cites_all_gold",
            "mean_milliseconds",
        )
        print(f"\n{'system':<16}" + "".join(f"{k[:14]:>16}" for k in keys))
        for name, s in comparison["systems"].items():
            print(f"{name:<16}" + "".join(f"{s[k]:>16}" for k in keys))
    elif args.command == "retrieval":
        from finagents.evaluation.runner import retrieval_recall

        print(json.dumps(retrieval_recall(args.split), indent=2))
    elif args.command == "research":
        from finagents.research import approvals

        if args.action == "run":
            from finagents.research.loop import ResearchLoop

            result = ResearchLoop(args.config, max_iterations=args.iterations).run()
            for step in result["trace"]:
                print(f"- {step['agent']}: {step['summary']}")
            print(
                f"\nsession {result['session']}: {result['iterations']} iteration(s), "
                f"candidates {result['candidates'] or 'none'}; log in docs/research-log.md"
            )
        elif args.action == "approvals":
            print(json.dumps(approvals.list_requests(), indent=2))
        else:
            decision = "approved" if args.action == "approve" else "rejected"
            print(json.dumps(approvals.decide(args.request_id, decision), indent=2))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("finagents.api.app:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
