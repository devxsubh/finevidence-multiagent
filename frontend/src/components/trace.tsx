"use client";

import { useState } from "react";
import { AGENT_NAMES, type TraceStep } from "@/lib/api";

// The execution trace: one numbered entry per agent run, with the orchestrator's routing
// decision after it. Retries show up as repeated agents, so the loop is visible.

export function Trace({ steps }: { steps: TraceStep[] }) {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <ol className="space-y-0">
      {steps.map((step, i) => {
        const failed = /not verified|insufficient|abstain|failed|no candidate|no calculation/.test(step.summary);
        const isOpen = open === i;
        return (
          <li key={i} className="relative pb-3 pl-8 last:pb-0">
            {i < steps.length - 1 && (
              <span className="absolute bottom-0 left-[11px] top-6 w-px bg-rule" aria-hidden />
            )}
            <span
              className={`absolute left-0 top-0.5 flex h-[23px] w-[23px] items-center justify-center rounded-full border text-[11px] font-medium ${
                failed ? "border-pencil text-pencil" : "border-rule-strong text-ink-soft"
              } bg-sheet`}
            >
              {i + 1}
            </span>
            <button
              onClick={() => setOpen(isOpen ? null : i)}
              aria-expanded={isOpen}
              className="w-full text-left"
            >
              <span className="flex items-baseline justify-between gap-2">
                <span className="text-[13px] font-medium text-ink">{AGENT_NAMES[step.agent] ?? step.agent}</span>
                <span className="shrink-0 text-[11px] text-ink-faint">{step.milliseconds.toFixed(1)} ms</span>
              </span>
              <span className={`mt-0.5 block text-[12px] leading-snug ${failed ? "text-pencil" : "text-ink-soft"}`}>
                {step.summary}
              </span>
              {step.route && (
                <span className="mt-1 block text-[11px] text-ink-faint">
                  Next: {AGENT_NAMES[step.route] ?? step.route}
                </span>
              )}
            </button>
            {isOpen && (
              <div className="mt-2 space-y-1 border border-rule bg-paper p-2">
                <p className="text-[11px] text-ink-faint">Tools: {step.tools.join(", ") || "none"}</p>
                <pre className="max-h-64 overflow-auto font-mono text-[11px] leading-snug text-ink-soft">
                  {JSON.stringify(step.details, null, 2)}
                </pre>
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
