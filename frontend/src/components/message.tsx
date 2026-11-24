"use client";

import { useState } from "react";
import { CalculationAndChecks } from "@/components/answer";
import { Trace } from "@/components/trace";
import { AGENT_NAMES, type AskResult, type TraceStep } from "@/lib/api";

export type AssistantTurn = {
  id: number;
  question: string;
  steps: TraceStep[];
  result?: AskResult;
  error?: string;
};

function matches(value: AskResult["answer"]["value"], reference: number | string) {
  if (value === null) return false;
  if (typeof value === "number" && typeof reference === "number") return Math.round(value * 1e5) / 1e5 === reference;
  return value === reference;
}

const REASONS: [RegExp, string][] = [
  [/^unsupported/, "The agents answer calculations: changes, percentage changes, ratios, totals, averages, comparisons, and a row's highest or lowest value. This question doesn't ask for one of those."],
  [/^insufficient evidence/, "The report doesn't seem to contain the figures this question needs."],
  [/no calculation fits/, "The agents found figures, but none fit the calculation this question needs."],
  [/^ambiguous/, "Two different answers are equally well supported by the report, so the agents won't pick one."],
  [/^conflicting evidence/, "The report shows different values for the same figure, so the agents won't pick one."],
  [/^confidence|low confidence/, "The best answer found isn't well enough supported to give."],
  [/passed verification/, "Every calculation the agents tried failed a check."],
];

/** Plain-language reason for an abstention; the system's own reason stays visible below it. */
function explainAbstention(reason: string | null): string {
  return REASONS.find(([pattern]) => pattern.test(reason ?? ""))?.[1] ?? "The agents could not verify an answer.";
}

export function UserMessage({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <p className="max-w-[85%] bg-ink px-3.5 py-2.5 text-[14px] leading-relaxed text-sheet">{text}</p>
    </div>
  );
}

export function AssistantMessage({
  turn,
  selected,
  onSelect,
}: {
  turn: AssistantTurn;
  selected: boolean;
  onSelect: () => void;
}) {
  const [open, setOpen] = useState<"none" | "checks" | "trace">("none");
  const { result, steps, error } = turn;

  if (error) {
    return (
      <div role="alert" className="border-l-[3px] border-pencil bg-pencil-wash px-4 py-3 text-[13px] text-pencil">
        {error}
      </div>
    );
  }

  if (!result) {
    const last = steps[steps.length - 1];
    return (
      <div className="border-l-[3px] border-rule-strong bg-sheet px-4 py-3" aria-live="polite">
        <p className="text-[13px] font-medium text-ink">Agents working</p>
        <ol className="mt-2 space-y-1">
          {steps.map((step, i) => (
            <li key={i} className={`text-[12px] leading-snug ${step === last ? "text-ink" : "text-ink-faint"}`}>
              <span className="font-medium">{AGENT_NAMES[step.agent] ?? step.agent}</span>: {step.summary}
            </li>
          ))}
          {steps.length === 0 && <li className="text-[12px] text-ink-faint">Starting…</li>}
        </ol>
      </div>
    );
  }

  const a = result.answer;
  const answered = a.status === "answered";
  const used = a.citations.flatMap((c) => c.values);
  return (
    <article
      onClick={onSelect}
      className={`cursor-default border-l-[3px] bg-sheet px-4 py-3.5 ${
        selected ? (answered ? "border-ledger" : "border-pencil") : "border-rule"
      }`}
      aria-label={`Answer to: ${turn.question}`}
    >
      {answered ? (
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <p className="text-[30px] font-semibold leading-none tracking-tight text-ink">{a.display}</p>
          <p className="text-[12px] text-ink-soft">Verified, confidence {a.confidence.toFixed(2)}</p>
        </div>
      ) : (
        <p className="text-[17px] font-semibold text-pencil">No answer given</p>
      )}
      <p className="mt-2 text-[14px] leading-relaxed text-ink">{answered ? a.explanation : explainAbstention(a.abstention_reason)}</p>
      {!answered && <p className="mt-1 text-[12px] text-ink-faint">Reason recorded by the orchestrator: {a.abstention_reason}</p>}
      {a.formula && <p className="mt-1.5 font-mono text-[12px] text-ink-soft">{a.formula}</p>}
      {!answered && a.withheld_value !== null && (
        <p className="mt-1.5 text-[12px] text-ink-soft">
          Withheld best candidate: <span className="font-mono">{String(a.withheld_value)}</span>
        </p>
      )}

      {used.length > 0 && (
        <ul className="mt-3 space-y-0.5">
          {used.map((v) => (
            <li key={v.id} className="text-[12px] text-ink-soft">
              <span className="mr-1.5 font-semibold text-pencil">{v.id}</span>
              {v.label} {v.periods.length ? `(${v.periods.join(", ")})` : v.column ? `(${v.column})` : ""}:{" "}
              <span className="text-ink">{v.literal}</span> <span className="text-ink-faint">{v.unit !== "number" ? v.unit : ""} in {v.segment_key}</span>
            </li>
          ))}
        </ul>
      )}

      {result.reference && (
        <p className="mt-3 text-[12px] text-ink-soft">
          FinQA reference (scored value): <span className="text-ink">{String(result.reference.exe_ans)}</span>
          {answered &&
            (matches(a.value, result.reference.exe_ans) ? (
              <span className="ml-2 font-medium text-ledger">matches</span>
            ) : (
              <span className="ml-2 font-medium text-pencil">does not match</span>
            ))}
        </p>
      )}

      <div className="mt-3 flex flex-wrap gap-4 border-t border-rule pt-2.5 text-[12px]">
        {a.checks.length > 0 && (
          <button
            onClick={() => setOpen(open === "checks" ? "none" : "checks")}
            aria-expanded={open === "checks"}
            className="text-ink-soft hover:text-ink"
          >
            {a.checks.filter((c) => c.passed).length} of {a.checks.length} checks passed
          </button>
        )}
        <button
          onClick={() => setOpen(open === "trace" ? "none" : "trace")}
          aria-expanded={open === "trace"}
          className="text-ink-soft hover:text-ink"
        >
          Agent trace: {result.trace.length} steps, {result.milliseconds.toFixed(0)} ms
        </button>
        {!selected && (
          <button onClick={onSelect} className="text-ink-soft hover:text-ink">
            Show in report
          </button>
        )}
      </div>
      {open === "checks" && (
        <div className="mt-3">
          <CalculationAndChecks result={result} />
        </div>
      )}
      {open === "trace" && (
        <div className="mt-3">
          <Trace steps={result.trace} />
        </div>
      )}
    </article>
  );
}
