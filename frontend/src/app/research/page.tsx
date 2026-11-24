"use client";

import { useCallback, useEffect, useState } from "react";
import {
  api,
  type ApprovalRequest,
  type Hypothesis,
  type ResearchEntry,
  type ResearchStatus,
} from "@/lib/api";

const AGENTS = [
  ["Research analyst", "reads the verified literature notes and current results, names the biggest weakness"],
  ["Dataset agent", "profiles train questions: operation types, evidence sources, program lengths"],
  ["Experiment designer", "picks a hypothesis for the weakness and fixes the acceptance test in advance"],
  ["ML engineer", "builds the variant as a validated configuration; saves accepted candidates"],
  ["Evaluation agent", "runs control and variant on the same train questions"],
  ["Research critic", "checks leakage, exact McNemar significance and a second train slice"],
  ["Documentation agent", "writes every iteration to docs/research-log.md"],
];

function outcomeClass(outcome: string) {
  if (outcome.startsWith("accepted")) return "text-ledger";
  if (outcome.startsWith("awaiting")) return "text-ink";
  return "text-pencil";
}

export default function ResearchPage() {
  const [status, setStatus] = useState<ResearchStatus | null>(null);
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [requests, setRequests] = useState<ApprovalRequest[]>([]);
  const [sessions, setSessions] = useState<{ session: string; entries: ResearchEntry[] }[]>([]);
  const [iterations, setIterations] = useState(3);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    Promise.all([api.researchStatus(), api.hypotheses(), api.approvals(), api.sessions()])
      .then(([s, h, a, x]) => {
        setStatus(s);
        setHypotheses(h);
        setRequests(a);
        setSessions(x);
        setError(null);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (!status?.running) return;
    const timer = setInterval(refresh, 2000);
    return () => clearInterval(timer);
  }, [status?.running, refresh]);

  const start = () =>
    api
      .startResearch("full", iterations)
      .then(refresh)
      .catch((e: Error) => setError(e.message));

  const decide = (id: string, decision: "approved" | "rejected") =>
    api.decide(id, decision).then(refresh).catch((e: Error) => setError(e.message));

  const pending = requests.filter((r) => r.status === "pending");

  return (
    <div className="mx-auto w-full max-w-[1100px] space-y-10 px-6 py-8">
      <header className="grid gap-8 md:grid-cols-[minmax(0,1fr)_320px]">
        <div className="max-w-[70ch]">
          <h1 className="text-xl font-semibold tracking-tight">Research lab</h1>
          <p className="mt-2 text-[14px] leading-relaxed text-ink-soft">
            A second team of agents experiments on the question-answering system itself. Each iteration names a
            weakness, tests one pre-implemented change against the current system on train questions, and keeps it
            only if the critic finds a significant improvement that holds on a second train slice. Dev and test are
            never used here. Changes with an external cost or that touch the evaluation protocol wait for your approval.
          </p>
          <ol className="mt-4 grid gap-x-6 gap-y-1.5 text-[13px] sm:grid-cols-2">
            {AGENTS.map(([name, job], i) => (
              <li key={name} className="flex gap-2">
                <span className="w-4 shrink-0 text-right text-ink-faint">{i + 1}</span>
                <span>
                  <span className="font-medium text-ink">{name}</span>{" "}
                  <span className="text-ink-soft">{job}</span>
                </span>
              </li>
            ))}
          </ol>
        </div>
        <div className="self-start border border-rule bg-sheet p-4">
          <h2 className="text-sm font-medium">Run a research session</h2>
          <label className="mt-3 block">
            <span className="mb-1 block text-xs text-ink-soft">Iterations</span>
            <input
              type="number"
              min={1}
              max={9}
              value={iterations}
              onChange={(e) => setIterations(Math.max(1, Math.min(9, Number(e.target.value) || 1)))}
              className="w-24 border border-rule-strong bg-sheet px-2 py-1.5 text-sm"
            />
          </label>
          <button
            onClick={start}
            disabled={!status || status.running}
            className="mt-3 w-full bg-ink px-3 py-2 text-sm font-medium text-sheet disabled:opacity-50"
          >
            {status?.running ? "Research running…" : "Start research session"}
          </button>
          <p className="mt-2 text-[11px] leading-snug text-ink-faint">
            Starts from the full system. About 10 seconds per iteration on CPU.
          </p>
          {status?.running && status.log.length > 0 && (
            <p className="mt-3 text-[12px] text-ink-soft" role="status">
              {status.log[status.log.length - 1]}
            </p>
          )}
          {status?.error && <p className="mt-3 text-[12px] text-pencil">{status.error}</p>}
        </div>
      </header>

      {error && (
        <p role="alert" className="border-l-[3px] border-pencil bg-pencil-wash px-3 py-2 text-[13px] text-pencil">
          {error}
        </p>
      )}

      <section>
        <h2 className="mb-1 text-base font-semibold">Waiting for your approval</h2>
        {pending.length === 0 ? (
          <p className="text-[13px] text-ink-faint">Nothing is waiting. Protected hypotheses appear here when a session reaches them.</p>
        ) : (
          <ul className="divide-y divide-rule border border-rule bg-sheet">
            {pending.map((r) => (
              <li key={r.id} className="flex flex-wrap items-center gap-4 px-4 py-3">
                <div className="min-w-0 flex-1">
                  <p className="text-[13px] font-medium text-ink">{r.subject}</p>
                  <p className="text-[12px] text-ink-soft">Needs approval because it {r.reason}.</p>
                </div>
                <button onClick={() => decide(r.id, "approved")} className="bg-ledger px-3 py-1.5 text-[13px] font-medium text-sheet">
                  Approve
                </button>
                <button onClick={() => decide(r.id, "rejected")} className="border border-rule-strong px-3 py-1.5 text-[13px] text-ink">
                  Reject
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h2 className="mb-3 text-base font-semibold">Sessions</h2>
        {sessions.length === 0 ? (
          <p className="text-[13px] text-ink-faint">No session has run yet. Start one above.</p>
        ) : (
          <div className="space-y-6">
            {sessions.map((s) => (
              <div key={s.session}>
                <h3 className="mb-2 text-sm font-medium text-ink">{s.session}</h3>
                <ol className="divide-y divide-rule border border-rule bg-sheet">
                  {s.entries.map((e) => (
                    <li key={e.iteration} className="grid gap-x-4 gap-y-1 px-4 py-3 text-[13px] md:grid-cols-[32px_minmax(0,1fr)_200px]">
                      <span className="text-ink-faint">{e.iteration}</span>
                      <div className="min-w-0">
                        <p>
                          <span className="font-medium text-ink">{e.hypothesis ?? "No hypothesis"}</span>
                          {e.statement && <span className="text-ink-soft"> {e.statement}</span>}
                        </p>
                        {e.delta && (
                          <p className="mt-1 text-[12px] text-ink-soft">
                            {Object.entries(e.delta)
                              .map(([k, v]) => `${k.replaceAll("_", " ")} ${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)} pts`)
                              .join(", ")}
                          </p>
                        )}
                        {[...(e.problems ?? []), ...(e.notes ?? [])].map((n) => (
                          <p key={n} className="mt-0.5 text-[12px] text-ink-faint">
                            {n}
                          </p>
                        ))}
                      </div>
                      <p className={`text-[12px] font-medium ${outcomeClass(e.outcome)}`}>{e.outcome}</p>
                    </li>
                  ))}
                </ol>
              </div>
            ))}
          </div>
        )}
      </section>

      <section>
        <h2 className="mb-1 text-base font-semibold">Hypotheses the team may test</h2>
        <p className="mb-3 text-[13px] text-ink-soft">
          Every hypothesis is a bounded, pre-implemented change. Agents cannot change the data, the split or the metric.
        </p>
        <div className="overflow-x-auto border border-rule bg-sheet">
          <table className="w-full border-collapse text-[13px]">
            <thead className="border-b border-rule bg-paper text-left">
              <tr>
                {["Hypothesis", "Addresses", "Change", "Must improve", "Approval"].map((h) => (
                  <th key={h} className="px-3 py-2 text-xs font-medium text-ink-soft">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {hypotheses.map((h) => (
                <tr key={h.id} className="border-b border-rule align-top last:border-b-0">
                  <td className="px-3 py-2">
                    <p className="font-medium text-ink">{h.id}</p>
                    <p className="text-[12px] text-ink-soft">{h.statement}</p>
                  </td>
                  <td className="px-3 py-2 text-ink-soft">{h.weakness.replaceAll("_", " ")}</td>
                  <td className="px-3 py-2 font-mono text-[11px] text-ink-soft">{JSON.stringify(h.change)}</td>
                  <td className="px-3 py-2 text-ink-soft">{h.objective.replaceAll("_", " ")}</td>
                  <td className="px-3 py-2">
                    {h.protected ? (
                      <span className={h.approval === "approved" ? "text-ledger" : "text-pencil"}>
                        {h.approval === "approved" ? "Approved" : h.approval === "rejected" ? "Rejected" : h.approval === "pending" ? "Waiting" : "Required"}
                      </span>
                    ) : (
                      <span className="text-ink-faint">Not needed</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
