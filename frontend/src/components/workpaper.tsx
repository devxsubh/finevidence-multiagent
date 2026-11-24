"use client";

import { useState } from "react";
import type { AskResult, EvidenceValue, Report } from "@/lib/api";

// The report drawn as an auditor's workpaper: rows the researcher retrieved carry a margin
// rule, and every figure the calculation used is highlighted, ticked and cross-referenced
// with its evidence id, the way an auditor marks figures they have checked.

function cellNumber(raw: string): number | null {
  const cleaned = raw.replace(/\(\s*[\d.,]+\s*%?\s*\)\s*$/, "").replace(/[$,%\s]/g, "");
  const negative = /^\(.*\)$/.test(raw.trim());
  const value = parseFloat(cleaned.replace(/[()]/g, ""));
  if (Number.isNaN(value)) return null;
  return negative ? -value : value;
}

function Tick({ id }: { id: string }) {
  return (
    <span className="ml-1 inline-flex items-baseline gap-0.5 align-super text-[10px] font-semibold text-pencil">
      <svg width="9" height="9" viewBox="0 0 10 10" aria-hidden className="translate-y-[1px]">
        <path d="M1 5.5 L4 8.5 L9 1.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
      {id}
    </span>
  );
}

export function Workpaper({ report, result }: { report: Report; result: AskResult | null }) {
  const [allText, setAllText] = useState(false);
  const retrieved = new Map((result?.retrieved ?? []).map((r) => [r.key, r.rank]));
  const used: EvidenceValue[] = result?.answer.citations.flatMap((c) => c.values) ?? [];
  const goldKeys = new Set(result?.reference?.gold_inds ?? []);
  const header = report.table[0] ?? [];

  const cellTag = (rowIndex: number, colIndex: number, raw: string): EvidenceValue | undefined =>
    used.find((v) => {
      if (v.segment_key !== `table_${rowIndex}`) return false;
      if (v.column && header[colIndex] !== v.column && rowIndex !== 0) return false;
      const number = cellNumber(raw);
      return number !== null && Math.abs(Math.abs(number) - Math.abs(parseFloat(v.literal))) < 1e-9;
    });

  const sentences = [...report.pre_text, ...report.post_text].map((text, i) => ({ key: `text_${i}`, text }));
  const relevant = sentences.filter((s) => retrieved.has(s.key) || used.some((v) => v.segment_key === s.key));
  const shown = allText || !result ? sentences : relevant;

  return (
    <section aria-label="Report" className="space-y-5">
      {report.table.length > 0 && (
        <div className="overflow-x-auto border border-rule bg-sheet">
          <table className="w-full border-collapse text-[13px]">
            <tbody>
              {report.table.map((row, r) => {
                const key = `table_${r}`;
                const isHeader = r === 0;
                const rank = retrieved.get(key);
                return (
                  <tr
                    key={key}
                    className={`border-b border-rule last:border-b-0 ${isHeader ? "bg-paper font-medium text-ink-soft" : ""}`}
                  >
                    <td
                      className={`w-20 border-r border-rule px-2 py-1.5 align-top text-[11px] text-ink-faint ${
                        rank ? "border-l-[3px] border-l-ledger" : "border-l-[3px] border-l-transparent"
                      }`}
                      title={rank ? `retrieved, rank ${rank}` : undefined}
                    >
                      {key}
                      {goldKeys.has(key) && <span className="ml-1 text-ledger" title="FinQA gold evidence">●</span>}
                    </td>
                    {row.map((cell, c) => {
                      const tag = cellTag(r, c, cell);
                      return (
                        <td
                          key={c}
                          className={`px-3 py-1.5 align-top ${c === 0 ? "text-left" : "text-right whitespace-nowrap"} ${
                            tag ? "bg-highlight/70 font-medium" : ""
                          }`}
                        >
                          {cell || <span className="text-ink-faint">–</span>}
                          {tag && <Tick id={tag.id} />}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {sentences.length > 0 && (
        <div>
          <div className="mb-2 flex items-baseline justify-between">
            <h3 className="text-sm font-medium text-ink">Report text</h3>
            {result && sentences.length > relevant.length && (
              <button onClick={() => setAllText(!allText)} className="text-xs text-ink-soft underline-offset-2 hover:underline">
                {allText ? `Show only the ${relevant.length} retrieved sentences` : `Show all ${sentences.length} sentences`}
              </button>
            )}
          </div>
          {shown.length === 0 ? (
            <p className="text-sm text-ink-faint">No sentence was retrieved for this question.</p>
          ) : (
            <ol className="space-y-1.5 text-[13px] leading-relaxed">
              {shown.map((s) => {
                const values = used.filter((v) => v.segment_key === s.key);
                const rank = retrieved.get(s.key);
                return (
                  <li
                    key={s.key}
                    className={`flex gap-3 border-l-[3px] pl-3 ${rank ? "border-l-ledger" : "border-l-transparent"}`}
                  >
                    <span className="w-14 shrink-0 text-[11px] text-ink-faint">
                      {s.key}
                      {goldKeys.has(s.key) && <span className="ml-1 text-ledger">●</span>}
                    </span>
                    <span className={values.length ? "bg-highlight/70" : "text-ink-soft"}>
                      {s.text}
                      {values.map((v) => (
                        <Tick key={v.id} id={v.id} />
                      ))}
                    </span>
                  </li>
                );
              })}
            </ol>
          )}
        </div>
      )}

      {result && (
        <p className="flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-ink-faint">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-[3px] bg-ledger" /> retrieved by the evidence researcher
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-4 bg-highlight" /> figure used in the calculation, with its evidence id
          </span>
          {result.reference && (
            <span className="flex items-center gap-1.5">
              <span className="text-ledger">●</span> FinQA gold evidence (not shown to the agents)
            </span>
          )}
        </p>
      )}
    </section>
  );
}
