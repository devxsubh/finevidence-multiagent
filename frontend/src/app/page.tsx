"use client";

import { useEffect, useRef, useState } from "react";
import { AssistantMessage, type AssistantTurn, UserMessage } from "@/components/message";
import { ReportPicker, type ReportSource } from "@/components/report-picker";
import { Workpaper } from "@/components/workpaper";
import { api, askStream, type AskBody, type Report } from "@/lib/api";

/** "candidates/R20261124-163148-H1-confidence-up" -> "candidate H1 (session 163148)" */
function configLabel(name: string) {
  const match = name.match(/^candidates\/R\d+-(\d+)-(H\d+)/);
  return match ? `candidate ${match[2]} (session ${match[1]})` : name;
}

// The assistant: a conversation about one report. Each question runs the eight agents with
// live progress; each reply carries its answer, evidence, checks and trace. The report sits
// beside the chat as a workpaper with the selected answer's figures ticked.

export default function AssistantPage() {
  const [report, setReport] = useState<Report | null>(null);
  const [source, setSource] = useState<ReportSource | null>(null);
  const [turns, setTurns] = useState<AssistantTurn[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const [picking, setPicking] = useState(false);
  const [configs, setConfigs] = useState<string[]>(["full"]);
  const [config, setConfig] = useState("full");
  const thread = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);
  const busy = turns.some((t) => !t.result && !t.error);

  useEffect(() => {
    api
      .health()
      .then((h) => {
        setConfigs(h.configs);
        // With an API key configured, chat with the Claude-assisted system by default.
        if (h.llm.installed && h.llm.credentials && h.configs.includes("full_llm")) setConfig("full_llm");
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    thread.current?.scrollTo({ top: thread.current.scrollHeight });
  }, [turns]);

  const pick = (picked: Report, from: ReportSource) => {
    setReport(picked);
    setSource(from);
    setTurns([]);
    setSelected(null);
    setDraft(from.kind === "dataset" ? from.suggestion : "");
    setPicking(false);
  };

  const ask = async (question: string) => {
    if (!report || !question.trim() || busy) return;
    const id = nextId.current++;
    setTurns((all) => [...all, { id, question: question.trim(), steps: [] }]);
    setDraft("");
    const body: AskBody =
      source?.kind === "dataset"
        ? { config, split: "dev", index: source.index, question }
        : { config, record: { question, table: report.table, pre_text: report.pre_text, post_text: report.post_text } };
    const update = (change: Partial<AssistantTurn>) =>
      setTurns((all) => all.map((t) => (t.id === id ? { ...t, ...change } : t)));
    try {
      const result = await askStream(body, (step) =>
        setTurns((all) => all.map((t) => (t.id === id ? { ...t, steps: [...t.steps, step] } : t))),
      );
      update({ result });
      setSelected(id);
    } catch (e) {
      update({ error: (e as Error).message });
    }
  };

  const shown = turns.find((t) => t.id === selected)?.result ?? null;
  const suggestion = source?.kind === "dataset" && !turns.some((t) => t.question === source.suggestion) ? source.suggestion : null;

  return (
    <div className="mx-auto grid w-full max-w-[1500px] flex-1 grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
      {/* Conversation */}
      <section className="flex min-h-[70vh] flex-col border-rule bg-paper xl:h-[calc(100vh-57px)] xl:border-r">
        <div className="flex flex-wrap items-center gap-3 border-b border-rule bg-sheet px-5 py-2.5">
          <div className="min-w-0 flex-1">
            <p className="truncate text-[13px] font-medium text-ink">{report ? report.id : "No report open"}</p>
            <p className="text-[11px] text-ink-faint">
              {report ? `${report.table.length} table rows, ${report.pre_text.length + report.post_text.length} sentences` : "Choose a report to start"}
            </p>
          </div>
          <label className="flex items-center gap-1.5 text-[12px] text-ink-soft">
            System
            <select value={config} onChange={(e) => setConfig(e.target.value)} className="max-w-[190px] border border-rule-strong bg-sheet px-1.5 py-1 text-[12px]">
              {configs.map((c) => (
                <option key={c} value={c}>
                  {configLabel(c)}
                </option>
              ))}
            </select>
          </label>
          <button onClick={() => setPicking(true)} className="border border-rule-strong bg-sheet px-3 py-1.5 text-[12px] font-medium text-ink hover:bg-paper">
            {report ? "Change report" : "Choose a report"}
          </button>
        </div>

        <div ref={thread} className="flex-1 space-y-4 overflow-y-auto px-5 py-5">
          {!report ? (
            <div className="mx-auto max-w-[52ch] pt-[10vh]">
              <h1 className="text-2xl font-semibold tracking-tight text-ink">Ask questions about a financial report</h1>
              <p className="mt-3 text-[15px] leading-relaxed text-ink-soft">
                Open a report, then ask in plain words: changes between years, ratios, totals, averages. Eight agents
                find the figures, check them, calculate on a deterministic calculator and verify the result. Every
                answer shows its evidence; when the evidence is not good enough, they say so instead of guessing.
              </p>
              <button onClick={() => setPicking(true)} className="mt-6 bg-ink px-4 py-2.5 text-sm font-medium text-sheet">
                Choose a report
              </button>
            </div>
          ) : turns.length === 0 ? (
            <p className="pt-6 text-center text-[13px] text-ink-faint">
              Ask anything about this report. Changing the report starts a new conversation.
            </p>
          ) : (
            turns.map((turn) => (
              <div key={turn.id} className="space-y-2.5">
                <UserMessage text={turn.question} />
                <AssistantMessage turn={turn} selected={selected === turn.id} onSelect={() => setSelected(turn.id)} />
              </div>
            ))
          )}
        </div>

        {report && (
          <form
            className="border-t border-rule bg-sheet px-5 py-3"
            onSubmit={(e) => {
              e.preventDefault();
              ask(draft);
            }}
          >
            {suggestion && (
              <button
                type="button"
                onClick={() => ask(suggestion)}
                disabled={busy}
                className="mb-2 block max-w-full truncate border border-rule px-2.5 py-1 text-left text-[12px] text-ink-soft hover:border-rule-strong hover:text-ink disabled:opacity-50"
              >
                Ask the FinQA question: {suggestion}
              </button>
            )}
            <div className="flex items-end gap-2">
              <label className="sr-only" htmlFor="question">
                Your question
              </label>
              <textarea
                id="question"
                rows={2}
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    ask(draft);
                  }
                }}
                placeholder="Ask about this report, e.g. what was the percentage change in revenue from 2018 to 2019?"
                className="min-h-[44px] flex-1 resize-none border border-rule-strong bg-sheet px-3 py-2 text-[14px] leading-snug"
              />
              <button
                type="submit"
                disabled={busy || !draft.trim()}
                className="h-[44px] bg-ink px-4 text-sm font-medium text-sheet disabled:opacity-50"
              >
                {busy ? "Working…" : "Ask"}
              </button>
            </div>
            <p className="mt-1.5 text-[11px] text-ink-faint">Enter to send, Shift + Enter for a new line. Not financial advice.</p>
          </form>
        )}
      </section>

      {/* Workpaper */}
      <section className="min-w-0 border-t border-rule px-5 py-5 xl:h-[calc(100vh-57px)] xl:overflow-y-auto xl:border-t-0 xl:px-8">
        {report ? (
          <>
            <div className="mb-4">
              <h2 className="text-sm font-medium text-ink">Report</h2>
              <p className="text-[12px] text-ink-faint">
                {!shown
                  ? "Ask a question to see which figures the agents use."
                  : shown.answer.citations.length
                    ? `Figures used to answer "${shown.question}" are ticked with their evidence id.`
                    : `No figures were used for "${shown.question}": the agents did not answer it.`}
              </p>
            </div>
            <Workpaper report={report} result={shown} />
          </>
        ) : (
          <p className="pt-[12vh] text-center text-[13px] text-ink-faint">The open report appears here.</p>
        )}
      </section>

      {picking && <ReportPicker onPick={pick} onClose={() => setPicking(false)} />}
    </div>
  );
}
