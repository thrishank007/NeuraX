"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  FileText,
  GitBranch,
  Library,
  MessageSquare,
  Moon,
  Plus,
  Search,
  Settings,
  Sun,
  Layers,
  Compass,
  Cpu,
} from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { StatusBar } from "@/components/layout/status-bar";
import { Button } from "@/components/ui/button";

const NAV = [
  { href: "/chat", label: "Search & Chat", icon: Compass },
  { href: "/search", label: "Multi-Search", icon: Search },
  { href: "/documents", label: "Library", icon: FileText },
  { href: "/sources", label: "Provenance", icon: Library },
  { href: "/graph", label: "Knowledge Graph", icon: GitBranch },
  { href: "/settings", label: "Diagnostics", icon: Settings },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [dark, setDark] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem("neurax-theme");
    const preferDark =
      stored === "dark" ||
      (!stored && window.matchMedia("(prefers-color-scheme: dark)").matches);
    setDark(preferDark);
    document.documentElement.classList.toggle("dark", preferDark);
  }, []);

  function toggleTheme() {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("neurax-theme", next ? "dark" : "light");
  }

  return (
    <div className="flex h-full min-h-screen bg-bg text-text selection:bg-accent-subtle">
      {/* Slim Perplexity-Style Left Rail */}
      <aside
        aria-label="Sidebar Navigation"
        className="hidden sm:flex flex-col justify-between w-16 shrink-0 border-r border-border bg-surface py-4 items-center z-30"
      >
        {/* Top: Brand & New Thread */}
        <div className="flex flex-col items-center gap-4 w-full">
          <Link
            href="/chat"
            className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent/15 border border-accent/30 text-accent font-mono text-sm font-bold shadow-xs hover:scale-105 transition-transform"
            title="NeuraX Home"
          >
            NX
          </Link>

          {/* New Search Button */}
          <button
            type="button"
            onClick={() => {
              if (pathname === "/chat") {
                window.location.href = "/chat";
              } else {
                router.push("/chat");
              }
            }}
            className="flex h-9 w-9 items-center justify-center rounded-xl bg-surface-2 hover:bg-surface-3 border border-border text-muted hover:text-text transition-colors cursor-pointer"
            title="New Search / Thread"
            aria-label="New Search"
          >
            <Plus className="h-4 w-4" />
          </button>

          <div className="w-8 h-px bg-border/60 my-1" />

          {/* Nav Items */}
          <nav className="flex flex-col gap-2">
            {NAV.map((item) => {
              const active =
                pathname === item.href || pathname.startsWith(`${item.href}/`);
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex h-9 w-9 items-center justify-center rounded-xl transition-all relative group cursor-pointer",
                    active
                      ? "bg-accent/15 text-accent border border-accent/20"
                      : "text-muted hover:bg-surface-2 hover:text-text",
                  )}
                  title={item.label}
                  aria-label={item.label}
                >
                  <Icon className="h-4 w-4" aria-hidden />

                  {/* Tooltip on hover */}
                  <span className="absolute left-14 whitespace-nowrap rounded-md bg-surface-3 px-2.5 py-1 text-xs font-medium text-text border border-border opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity z-50 shadow-hud">
                    {item.label}
                  </span>
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Bottom: Theme Toggle & Air-Gap status */}
        <div className="flex flex-col items-center gap-3">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={toggleTheme}
            aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
            className="h-8 w-8 text-muted hover:text-text rounded-xl"
          >
            {dark ? (
              <Sun className="h-4 w-4 text-signal-amber" aria-hidden />
            ) : (
              <Moon className="h-4 w-4 text-signal-cyan" aria-hidden />
            )}
          </Button>

          <div
            className="flex h-7 w-7 items-center justify-center rounded-full bg-surface-2 border border-border text-[10px] text-accent font-mono font-bold"
            title="Local Air-Gapped Intelligence"
          >
            <span className="h-2 w-2 rounded-full bg-accent animate-pulse-dot" />
          </div>
        </div>
      </aside>

      {/* Main Canvas Area */}
      <div className="flex flex-1 flex-col min-w-0 overflow-hidden">
        {/* Mobile Header */}
        <header className="flex h-12 items-center justify-between border-b border-border bg-surface px-4 sm:hidden">
          <Link href="/chat" className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-lg bg-accent/15 border border-accent/30 text-accent font-mono text-xs font-bold">
              NX
            </div>
            <span className="font-semibold tracking-tight text-sm">NeuraX</span>
          </Link>

          <div className="flex items-center gap-2">
            <StatusBar />
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={toggleTheme}
              className="h-7 w-7"
            >
              {dark ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
            </Button>
          </div>
        </header>

        {/* Workspace Canvas */}
        <main id="main" className="min-w-0 flex-1 overflow-auto scroll-panel bg-bg">
          {children}
        </main>

        {/* Mobile Bottom Navigation */}
        <nav
          aria-label="Mobile Navigation"
          className="flex border-t border-border bg-surface sm:hidden z-20"
        >
          {NAV.map((item) => {
            const active = pathname.startsWith(item.href);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex flex-1 flex-col items-center gap-1 py-2 text-[10px] font-medium transition-colors",
                  active ? "text-accent font-semibold" : "text-muted hover:text-text",
                )}
              >
                <Icon className="h-4 w-4" aria-hidden />
                <span>{item.label.split(" ")[0]}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </div>
  );
}
