"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  FileText,
  GitBranch,
  MessageSquare,
  PanelLeftClose,
  PanelLeftOpen,
  Settings,
  Library,
} from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { StatusBar } from "@/components/layout/status-bar";
import { Button } from "@/components/ui/button";

const NAV = [
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/documents", label: "Documents", icon: FileText },
  { href: "/sources", label: "Sources", icon: Library },
  { href: "/graph", label: "Knowledge Graph", icon: GitBranch },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [dark, setDark] = useState(false);

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
    <div className="flex h-full min-h-screen flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-accent focus:px-3 focus:py-2 focus:text-white"
      >
        Skip to main content
      </a>

      <header className="flex h-12 items-center justify-between border-b border-border bg-surface px-3">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}
            onClick={() => setCollapsed((c) => !c)}
          >
            {collapsed ? (
              <PanelLeftOpen className="h-4 w-4" />
            ) : (
              <PanelLeftClose className="h-4 w-4" />
            )}
          </Button>
          <Link href="/chat" className="font-semibold tracking-tight text-text">
            NeuraX
          </Link>
          <span className="hidden text-xs text-muted sm:inline">
            Local document intelligence
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={toggleTheme}>
            {dark ? "Light" : "Dark"}
          </Button>
        </div>
      </header>

      <StatusBar />

      <div className="flex min-h-0 flex-1">
        <nav
          aria-label="Primary"
          className={cn(
            "hidden shrink-0 border-r border-border bg-surface md:flex md:flex-col",
            collapsed ? "w-14" : "w-56",
          )}
        >
          <ul className="flex flex-col gap-1 p-2">
            {NAV.map((item) => {
              const active =
                pathname === item.href || pathname.startsWith(`${item.href}/`);
              const Icon = item.icon;
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    className={cn(
                      "flex items-center gap-2 rounded-md px-2 py-2 text-sm transition-colors",
                      active
                        ? "bg-accent-subtle text-accent"
                        : "text-muted hover:bg-surface-2 hover:text-text",
                    )}
                    title={item.label}
                  >
                    <Icon className="h-4 w-4 shrink-0" aria-hidden />
                    {!collapsed && <span>{item.label}</span>}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        <main id="main" className="min-w-0 flex-1 overflow-auto scroll-panel">
          {children}
        </main>
      </div>

      <nav
        aria-label="Mobile"
        className="flex border-t border-border bg-surface md:hidden"
      >
        {NAV.map((item) => {
          const active = pathname.startsWith(item.href);
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex flex-1 flex-col items-center gap-0.5 py-2 text-[10px]",
                active ? "text-accent" : "text-muted",
              )}
            >
              <Icon className="h-4 w-4" aria-hidden />
              {item.label.split(" ")[0]}
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
