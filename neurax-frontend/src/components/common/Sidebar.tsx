"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, 
  MessageSquare, 
  Files, 
  History, 
  BarChart3, 
  Settings,
  ShieldAlert
} from "lucide-react";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Query", href: "/", icon: MessageSquare },
  { name: "Documents", href: "/documents", icon: Files },
  { name: "History", href: "/history", icon: History },
  { name: "Analytics", href: "/analytics", icon: BarChart3 },
  { name: "Security", href: "/security", icon: ShieldAlert },
  { name: "Settings", href: "/settings", icon: Settings },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="flex h-full w-64 flex-col bg-slate-900 text-white">
      <div className="flex h-16 items-center px-6">
        <span className="text-2xl font-bold text-blue-400">NeuraX</span>
      </div>
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navigation.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "group flex items-center rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive 
                  ? "bg-slate-800 text-blue-400" 
                  : "text-slate-300 hover:bg-slate-800 hover:text-white"
              )}
            >
              <item.icon className={cn(
                "mr-3 h-5 w-5 flex-shrink-0",
                isActive ? "text-blue-400" : "text-slate-400 group-hover:text-white"
              )} />
              {item.name}
            </Link>
          );
        })}
      </nav>
      <div className="border-t border-slate-800 p-4">
        <div className="flex items-center">
          <div className="h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center text-xs font-bold">
            NX
          </div>
          <div className="ml-3">
            <p className="text-sm font-medium">System Admin</p>
            <p className="text-xs text-slate-400">Offline Mode</p>
          </div>
        </div>
      </div>
    </div>
  );
}
