// Typed client for the FinAgents API (see src/finagents/api/app.py).

export type Check = { name: string; passed: boolean; detail: string; hard: boolean };

export type EvidenceValue = {
  id: string;
  segment_key: string;
  source: "table" | "text";
  label: string;
  column: string;
  periods: string[];
  literal: string;
  value: number;
  unit: string;
  relevance: number;
};

export type Citation = { segment_key: string; source: string; text: string; values: EvidenceValue[] };

export type Answer = {
  status: "answered" | "abstained";
  value: number | string | null;
  display: string | null;
  explanation: string;
  citations: Citation[];
  calculation: string[];
  formula: string | null;
  program: string | null;
  confidence: number;
  abstention_reason: string | null;
  checks: Check[];
  alternatives: string[];
  withheld_value: number | string | null;
};

export type TraceStep = {
  agent: string;
  summary: string;
  tools: string[];
  route: string | null;
  milliseconds: number;
  details: Record<string, unknown>;
};

export type Report = {
  id: string;
  question: string;
  table: string[][];
  pre_text: string[];
  post_text: string[];
};

export type AskResult = {
  id: string;
  question: string;
  config: string;
  answer: Answer;
  analysis: { metric: string; operation: string; periods: string[]; backend: string; entity_terms: string[] };
  trace: TraceStep[];
  retrieved: { key: string; rank: number }[];
  evidence: EvidenceValue[];
  research_rounds: number;
  candidates_tried: number;
  milliseconds: number;
  report: Report;
  reference?: { exe_ans: number | string; answer: string; program: string; gold_inds: string[] };
};

export type Health = {
  status: string;
  llm: { provider: string; installed: boolean; credentials: boolean };
  reranker_trained: boolean;
  configs: string[];
};

export type SystemSummary = {
  config: string;
  settings: { description: string };
  questions: number;
  execution_accuracy: number;
  execution_accuracy_ci95: [number, number];
  coverage: number;
  selective_accuracy: number;
  wrong_answer_rate: number;
  cites_all_gold: number;
  withheld_but_correct: number;
  abstentions: Record<string, number>;
  mean_rounds: number;
  mean_milliseconds: number;
  by_operation: Record<string, { questions: number; execution_accuracy: number; coverage: number; wrong_answer_rate: number }>;
  vs_baseline?: {
    correct: { right_only: number; left_only: number; p_value: number };
    wrong_answers_avoided: { right_only: number; left_only: number; p_value: number };
  };
};

export type Evaluation = {
  comparison: { split: string; questions: number; baseline: string; systems: Record<string, SystemSummary> };
  retrieval: Record<string, Record<string, number> | number> | null;
};

export type Hypothesis = {
  id: string;
  weakness: string;
  statement: string;
  change: Record<string, unknown>;
  objective: string;
  guard: string;
  protected: boolean;
  approval: "pending" | "approved" | "rejected" | null;
};

export type ApprovalRequest = {
  id: string;
  action: string;
  subject: string;
  reason: string;
  status: "pending" | "approved" | "rejected";
  requested: string;
  decided: string | null;
};

export type ResearchEntry = {
  iteration: number;
  hypothesis?: string;
  statement?: string;
  weakness?: string;
  acceptance?: string;
  delta?: Record<string, number>;
  outcome: string;
  notes?: string[];
  problems?: string[];
  candidate?: string;
};

export type ResearchStatus = {
  running: boolean;
  error: string | null;
  log: string[];
  last: { session: string; iterations: number; candidates: string[]; trace: { agent: string; summary: string }[] } | null;
};

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(0, "The API is not reachable. Start it with: uv run finagents serve");
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // keep the status text
    }
    if (response.status >= 500) detail = "The API is not reachable. Start it with: uv run finagents serve";
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),
  examples: (q: string, offset: number, limit = 30) =>
    request<{ total: number; items: { index: number; id: string; question: string }[] }>(
      `/examples?split=dev&offset=${offset}&limit=${limit}&q=${encodeURIComponent(q)}`,
    ),
  example: (index: number) => request<Report & { index: number }>(`/examples/dev/${index}`),
  ask: (body: AskBody) =>
    request<AskResult>("/ask", { method: "POST", body: JSON.stringify(body) }),
  evaluation: () => request<Evaluation>("/evaluation"),
  hypotheses: () => request<Hypothesis[]>("/research/hypotheses"),
  approvals: () => request<ApprovalRequest[]>("/research/approvals"),
  decide: (id: string, decision: "approved" | "rejected") =>
    request<ApprovalRequest>(`/research/approvals/${id}`, { method: "POST", body: JSON.stringify({ decision }) }),
  sessions: () => request<{ session: string; entries: ResearchEntry[] }[]>("/research/sessions"),
  startResearch: (config: string, iterations: number) =>
    request<{ started: boolean }>("/research/run", { method: "POST", body: JSON.stringify({ config, iterations }) }),
  researchStatus: () => request<ResearchStatus>("/research/status"),
};

export type AskBody = {
  config: string;
  split?: string;
  index?: number;
  question?: string;
  record?: Omit<Report, "id">;
};

/** Ask with live progress: calls onStep for every agent as it finishes, resolves with the result. */
export async function askStream(body: AskBody, onStep: (step: TraceStep) => void): Promise<AskResult> {
  let response: Response;
  try {
    response = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "The API is not reachable. Start it with: uv run finagents serve");
  }
  if (!response.ok || !response.body) {
    let detail = response.status >= 500 ? "The API is not reachable. Start it with: uv run finagents serve" : `${response.status}`;
    try {
      const data = await response.json();
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      // keep the status
    }
    throw new ApiError(response.status, detail);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
      const event = block.match(/^event: (.*)$/m)?.[1];
      const data = block.match(/^data: (.*)$/m)?.[1];
      if (!event || !data) continue;
      const parsed = JSON.parse(data);
      if (event === "step") onStep(parsed as TraceStep);
      if (event === "result") return parsed as AskResult;
      if (event === "error") throw new ApiError(500, parsed.detail);
    }
  }
  throw new ApiError(0, "The answer stream ended early. Try again.");
}

export const AGENT_NAMES: Record<string, string> = {
  orchestrator: "Orchestrator",
  query_analyst: "Query analyst",
  evidence_researcher: "Evidence researcher",
  evidence_auditor: "Evidence auditor",
  reasoning: "Reasoning agent",
  program_validator: "Program validator",
  calculation: "Calculation agent",
  verification: "Verification agent",
  response: "Response agent",
  abstain: "Abstain",
};

export const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;
