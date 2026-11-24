"use client";

import { useEffect, useRef, useState } from "react";
import { api, type Report } from "@/lib/api";

// Choose the report the conversation is about: any FinQA dev report, or your own.

export type ReportSource =
  | { kind: "dataset"; index: number; suggestion: string }
  | { kind: "own" };

type Item = { index: number; id: string; question: string };

const EXAMPLE_TABLE = "( in millions ) | 2019 | 2018\nrevenue | $ 1200 | $ 1000\noperating expenses | $ 900 | $ 800";
const EXAMPLE_TEXT = "revenue increased due to higher volume in 2019 .";

function parseTable(text: string): string[][] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.split("|").map((cell) => cell.trim()));
}

export function ReportPicker({
  onPick,
  onClose,
}: {
  onPick: (report: Report, source: ReportSource) => void;
  onClose: () => void;
}) {
  const [tab, setTab] = useState<"dataset" | "own">("dataset");
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<Item[]>([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [table, setTable] = useState(EXAMPLE_TABLE);
  const [text, setText] = useState(EXAMPLE_TEXT);
  const [name, setName] = useState("My report");
  const dialog = useRef<HTMLDivElement>(null);
  const request = useRef(0);

  useEffect(() => {
    const id = ++request.current;
    const timer = setTimeout(() => {
      api
        .examples(query, 0, 40)
        .then((page) => {
          if (id !== request.current) return;
          setItems(page.items);
          setTotal(page.total);
          setError(null);
        })
        .catch((e: Error) => id === request.current && setError(e.message));
    }, 200);
    return () => clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    dialog.current?.querySelector<HTMLElement>("input")?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const pickDataset = (item: Item) =>
    api
      .example(item.index)
      .then((report) => onPick(report, { kind: "dataset", index: item.index, suggestion: item.question }))
      .catch((e: Error) => setError(e.message));

  const rows = parseTable(table);
  const widths = new Set(rows.map((row) => row.length));
  const sentences = text.split("\n").map((l) => l.trim()).filter(Boolean);
  const problem =
    !rows.length && !sentences.length
      ? "Add a table or some text."
      : widths.size > 1
        ? "Every table row needs the same number of cells."
        : null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-ink/30 p-4 pt-[8vh]" onClick={onClose}>
      <div
        ref={dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="picker-title"
        onClick={(e) => e.stopPropagation()}
        className="flex max-h-[80vh] w-full max-w-[720px] flex-col border border-rule-strong bg-sheet shadow-[0_12px_40px_rgba(24,38,58,0.18)]"
      >
        <div className="flex items-center justify-between border-b border-rule px-5 py-3">
          <h2 id="picker-title" className="text-[15px] font-semibold">
            Choose a report
          </h2>
          <button onClick={onClose} className="text-[13px] text-ink-soft hover:text-ink">
            Close
          </button>
        </div>
        <div className="flex border-b border-rule" role="tablist">
          {(["dataset", "own"] as const).map((t) => (
            <button
              key={t}
              role="tab"
              aria-selected={tab === t}
              onClick={() => setTab(t)}
              className={`border-b-2 px-5 py-2.5 text-[13px] ${
                tab === t ? "border-ink font-medium text-ink" : "border-transparent text-ink-soft hover:text-ink"
              }`}
            >
              {t === "dataset" ? "FinQA reports" : "Paste your own"}
            </button>
          ))}
        </div>

        {tab === "dataset" ? (
          <div className="flex min-h-0 flex-1 flex-col">
            <div className="px-5 py-3">
              <input
                type="search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search the 883 dev reports by their question"
                aria-label="Search reports"
                className="w-full border border-rule-strong bg-sheet px-2.5 py-2 text-[13px]"
              />
              <p className="mt-1.5 text-[11px] text-ink-faint">
                {error ?? `${total} reports. Each comes with the question its FinQA annotator asked; you can ask anything.`}
              </p>
            </div>
            <ul className="min-h-0 flex-1 overflow-y-auto border-t border-rule">
              {items.map((item) => (
                <li key={item.index}>
                  <button
                    onClick={() => pickDataset(item)}
                    className="block w-full border-b border-rule px-5 py-2.5 text-left text-[13px] leading-snug text-ink-soft hover:bg-paper hover:text-ink"
                  >
                    <span className="mr-2 text-[11px] text-ink-faint">{item.id}</span>
                    {item.question}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <form
            className="space-y-4 overflow-y-auto px-5 py-4"
            onSubmit={(e) => {
              e.preventDefault();
              if (problem) return;
              onPick({ id: name.trim() || "My report", question: "", table: rows, pre_text: sentences, post_text: [] }, { kind: "own" });
            }}
          >
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-ink-soft">Name</span>
              <input value={name} onChange={(e) => setName(e.target.value)} className="w-full border border-rule-strong px-2.5 py-2 text-sm" />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-ink-soft">Table: one row per line, cells separated by |, header first</span>
              <textarea value={table} onChange={(e) => setTable(e.target.value)} rows={6} className="w-full border border-rule-strong px-2.5 py-2 font-mono text-[12px]" />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-ink-soft">Report text: one sentence per line</span>
              <textarea value={text} onChange={(e) => setText(e.target.value)} rows={4} className="w-full border border-rule-strong px-2.5 py-2 text-[13px]" />
            </label>
            <p className="text-[11px] text-ink-faint">The prefilled figures are illustrative, not from a real report.</p>
            {problem && <p className="text-xs text-pencil">{problem}</p>}
            <button type="submit" disabled={problem !== null} className="bg-ink px-4 py-2 text-sm font-medium text-sheet disabled:opacity-50">
              Use this report
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
