"use client";

import { useEffect, useState } from "react";
import { api, type Evaluation, pct, type SystemSummary } from "@/lib/api";

const LABELS: Record<string, string> = {
  single_pass: "Single pass (ablation)",
  no_abstention: "No abstention (ablation)",
  full: "Full multi-agent system",
  full_bm25: "Full system, BM25 retrieval",
};

const label = (name: string) =>
  LABELS[name] ?? (name.startsWith("candidates/") ? `Research-loop candidate ${name.split("-").slice(-2).join("-")}` : name);

const p = (value: number) => (value < 0.001 ? value.toExponential(1) : value.toFixed(3));

function Th({ children, right = true }: { children: React.ReactNode; right?: boolean }) {
  return <th className={`px-3 py-2 text-xs font-medium text-ink-soft ${right ? "text-right" : "text-left"}`}>{children}</th>;
}

export default function EvaluationPage() {
  const [data, setData] = useState<Evaluation | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.evaluation().then(setData).catch((e: Error) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="mx-auto w-full max-w-[1100px] px-6 py-10">
        <h1 className="text-xl font-semibold">Evaluation</h1>
        <p className="mt-3 text-[15px] text-ink-soft">{error}</p>
      </div>
    );
  }
  if (!data) return <p className="mx-auto w-full max-w-[1100px] px-6 py-10 text-ink-faint">Loading results…</p>;

  const { comparison, retrieval } = data;
  const systems = Object.entries(comparison.systems) as [string, SystemSummary][];
  const full = comparison.systems["full"];
  const kinds = Array.from(new Set(systems.flatMap(([, s]) => Object.keys(s.abstentions ?? {}))));

  return (
    <div className="mx-auto w-full max-w-[1100px] space-y-10 px-6 py-8">
      <header className="max-w-[75ch]">
        <h1 className="text-xl font-semibold tracking-tight">Evaluation on FinQA dev</h1>
        <p className="mt-2 text-[14px] leading-relaxed text-ink-soft">
          {comparison.questions} dev questions, official FinQA execution accuracy (an abstention counts as not
          correct). Settings and thresholds were chosen on train; dev was run once per system. The official test split
          is not used. Every system runs offline with deterministic agent backends.
        </p>
      </header>

      <section>
        <h2 className="mb-3 text-base font-semibold">Systems</h2>
        <div className="overflow-x-auto border border-rule bg-sheet">
          <table className="w-full border-collapse text-[13px]">
            <thead className="border-b border-rule bg-paper">
              <tr>
                <Th right={false}>System</Th>
                <Th>Execution accuracy (95% CI)</Th>
                <Th>Coverage</Th>
                <Th>Accuracy when answering</Th>
                <Th>Wrong-answer rate</Th>
                <Th>Cites all gold evidence</Th>
                <Th>ms per question</Th>
              </tr>
            </thead>
            <tbody>
              {systems.map(([name, s]) => (
                <tr key={name} className={`border-b border-rule last:border-b-0 ${name === "full" ? "font-medium" : ""}`}>
                  <td className="px-3 py-2">{label(name)}</td>
                  <td className="px-3 py-2 text-right">
                    {pct(s.execution_accuracy)}{" "}
                    <span className="text-ink-faint">
                      [{pct(s.execution_accuracy_ci95[0])}, {pct(s.execution_accuracy_ci95[1])}]
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right">{pct(s.coverage)}</td>
                  <td className="px-3 py-2 text-right">{pct(s.selective_accuracy)}</td>
                  <td className={`px-3 py-2 text-right ${s.wrong_answer_rate < 0.3 ? "text-ledger" : "text-pencil"}`}>
                    {pct(s.wrong_answer_rate)}
                  </td>
                  <td className="px-3 py-2 text-right">{pct(s.cites_all_gold)}</td>
                  <td className="px-3 py-2 text-right">{s.mean_milliseconds.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <ul className="mt-3 space-y-1 text-[13px] text-ink-soft">
          {systems
            .filter(([, s]) => s.vs_baseline)
            .map(([name, s]) => (
              <li key={name}>
                {label(name)} vs {label(comparison.baseline).toLowerCase()}:{" "}
                {s.vs_baseline!.wrong_answers_avoided.right_only} wrong answers avoided,{" "}
                {s.vs_baseline!.wrong_answers_avoided.left_only} added (exact McNemar p ={" "}
                {p(s.vs_baseline!.wrong_answers_avoided.p_value)}); {s.vs_baseline!.correct.right_only} correct answers
                gained, {s.vs_baseline!.correct.left_only} lost (p = {p(s.vs_baseline!.correct.p_value)}).
              </li>
            ))}
        </ul>
      </section>

      <section>
        <h2 className="mb-1 text-base font-semibold">Why the systems did not answer</h2>
        <p className="mb-3 text-[13px] text-ink-soft">Questions abstained on, by the reason the orchestrator recorded.</p>
        <div className="overflow-x-auto border border-rule bg-sheet">
          <table className="w-full border-collapse text-[13px]">
            <thead className="border-b border-rule bg-paper">
              <tr>
                <Th right={false}>System</Th>
                {kinds.map((k) => (
                  <Th key={k}>{k.replaceAll("_", " ")}</Th>
                ))}
                <Th>Withheld but correct</Th>
              </tr>
            </thead>
            <tbody>
              {systems.map(([name, s]) => (
                <tr key={name} className="border-b border-rule last:border-b-0">
                  <td className="px-3 py-2">{label(name)}</td>
                  {kinds.map((k) => (
                    <td key={k} className="px-3 py-2 text-right">
                      {s.abstentions?.[k] ?? 0}
                    </td>
                  ))}
                  <td className="px-3 py-2 text-right">{s.withheld_but_correct}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="grid gap-10 md:grid-cols-2">
        {retrieval && (
          <section>
            <h2 className="mb-1 text-base font-semibold">Evidence retrieval</h2>
            <p className="mb-3 text-[13px] text-ink-soft">
              Share of {String(retrieval.questions)} dev questions whose gold evidence is all in the top k, ranking with the
              question alone. The learned reranker was trained and selected on train.
            </p>
            <table className="w-full border-collapse border border-rule bg-sheet text-[13px]">
              <thead className="border-b border-rule bg-paper">
                <tr>
                  <Th right={false}>Retriever</Th>
                  <Th>Top 3</Th>
                  <Th>Top 5</Th>
                  <Th>Top 10</Th>
                </tr>
              </thead>
              <tbody>
                {(["bm25", "tfidf", "learned"] as const).map((name) => {
                  const r = retrieval[name] as Record<string, number>;
                  return (
                    <tr key={name} className={`border-b border-rule ${name === "learned" ? "font-medium" : ""}`}>
                      <td className="px-3 py-2">{{ bm25: "BM25", tfidf: "TF-IDF (train IDF)", learned: "Learned reranker" }[name]}</td>
                      <td className="px-3 py-2 text-right">{pct(r["recall@3"])}</td>
                      <td className="px-3 py-2 text-right">{pct(r["recall@5"])}</td>
                      <td className="px-3 py-2 text-right">{pct(r["recall@10"])}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </section>
        )}
        {full && (
          <section>
            <h2 className="mb-1 text-base font-semibold">Full system by question type</h2>
            <p className="mb-3 text-[13px] text-ink-soft">The operation the query analyst assigned to each question.</p>
            <table className="w-full border-collapse border border-rule bg-sheet text-[13px]">
              <thead className="border-b border-rule bg-paper">
                <tr>
                  <Th right={false}>Operation</Th>
                  <Th>Questions</Th>
                  <Th>Exec. acc.</Th>
                  <Th>Coverage</Th>
                  <Th>Wrong</Th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(full.by_operation).map(([op, s]) => (
                  <tr key={op} className="border-b border-rule">
                    <td className="px-3 py-2">{op.replaceAll("_", " ")}</td>
                    <td className="px-3 py-2 text-right">{s.questions}</td>
                    <td className="px-3 py-2 text-right">{pct(s.execution_accuracy)}</td>
                    <td className="px-3 py-2 text-right">{pct(s.coverage)}</td>
                    <td className="px-3 py-2 text-right">{pct(s.wrong_answer_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
      </div>
    </div>
  );
}
