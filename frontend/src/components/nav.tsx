"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { api, type Health } from "@/lib/api";

const LINKS = [
  { href: "/", label: "Assistant" },
  { href: "/evaluation", label: "Evaluation" },
  { href: "/research", label: "Research lab" },
];

export function Nav() {
  const pathname = usePathname();
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setDown(true));
  }, []);

  const llm = health?.llm.installed && health.llm.credentials ? "Claude available" : "LLM off";

  return (
    <header className="border-b border-rule bg-sheet">
      <div className="mx-auto flex h-14 max-w-[1500px] items-center gap-4 px-4 sm:gap-8 sm:px-6">
        <Link href="/" className="text-[15px] font-semibold tracking-tight text-ink">
          FinAgents
        </Link>
        <nav aria-label="Main" className="flex h-full items-stretch gap-1">
          {LINKS.map((link) => {
            const active = link.href === "/" ? pathname === "/" : pathname.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                aria-current={active ? "page" : undefined}
                className={`flex items-center whitespace-nowrap border-b-2 px-2 text-sm sm:px-3 ${
                  active ? "border-ink font-medium text-ink" : "border-transparent text-ink-soft hover:text-ink"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto hidden items-center gap-4 text-xs text-ink-soft md:flex" role="status">
          {down ? (
            <span className="text-pencil">API offline: run uv run finagents serve</span>
          ) : health ? (
            <>
              <span className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-ledger" aria-hidden />
                API connected
              </span>
              <span>{health.reranker_trained ? "Learned retriever ready" : "Retriever not trained"}</span>
              <span>{llm}</span>
            </>
          ) : (
            <span>Connecting…</span>
          )}
        </div>
      </div>
    </header>
  );
}
