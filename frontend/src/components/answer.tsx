"use client";

import type { AskResult } from "@/lib/api";

export function CalculationAndChecks({ result }: { result: AskResult }) {
  const a = result.answer;
  if (!a.calculation.length && !a.checks.length) return null;
  return (
    <section aria-label="Calculation and verification" className="grid gap-8 md:grid-cols-2">
      <div>
        <h3 className="mb-2 text-sm font-medium text-ink">Calculation</h3>
        {a.program && <p className="mb-2 font-mono text-[12px] text-ink-soft">{a.program}</p>}
        <ol className="space-y-1 text-[13px]">
          {a.calculation.map((line) => (
            <li key={line} className="font-mono text-[12px] text-ink">
              {line}
            </li>
          ))}
        </ol>
        {a.alternatives.length > 0 && (
          <div className="mt-4">
            <h4 className="mb-1 text-xs font-medium text-ink-soft">Other candidates tried</h4>
            <ul className="space-y-0.5">
              {a.alternatives.map((alt) => (
                <li key={alt} className="font-mono text-[11px] text-ink-faint">
                  {alt}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <div>
        <h3 className="mb-2 text-sm font-medium text-ink">Verification checks</h3>
        <ul className="space-y-1.5">
          {a.checks.map((check) => (
            <li key={check.name} className="flex gap-2 text-[13px]">
              <span
                className={`mt-0.5 w-4 shrink-0 text-center font-semibold ${check.passed ? "text-ledger" : "text-pencil"}`}
                aria-label={check.passed ? "passed" : "failed"}
              >
                {check.passed ? "✓" : "✗"}
              </span>
              <span>
                <span className="text-ink">{check.name.replaceAll("_", " ")}</span>
                {!check.hard && <span className="text-ink-faint"> (lowers confidence)</span>}
                <span className="block text-[12px] text-ink-soft">{check.detail}</span>
              </span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
