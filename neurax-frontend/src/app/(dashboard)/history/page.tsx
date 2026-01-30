"use client";

import { useQueryStore } from "@/lib/store/queryStore";
import { MessageSquare, Clock, ArrowRight, Trash2 } from "lucide-react";

export default function HistoryPage() {
  const { history } = useQueryStore();

  if (history.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] text-center">
        <div className="p-4 bg-slate-100 dark:bg-slate-800 rounded-full text-slate-400 mb-4">
          <History size={48} />
        </div>
        <h2 className="text-xl font-bold text-slate-900 dark:text-white">No query history</h2>
        <p className="text-slate-500 dark:text-slate-400 mt-2 max-w-xs">
          Your previous searches and interactions will appear here.
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl space-y-8 pb-12">
      <div>
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white flex items-center">
          <Clock className="mr-3 text-slate-400" /> Query History
        </h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1">
          Review and manage your previous interactions
        </p>
      </div>

      <div className="space-y-4">
        {history.map((item, idx) => (
          <div 
            key={idx}
            className="group bg-white dark:bg-slate-900 border dark:border-slate-800 rounded-2xl p-6 shadow-sm hover:border-blue-300 dark:hover:border-blue-700 transition-all cursor-pointer"
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-4 flex-1">
                <div className="p-2 bg-blue-50 dark:bg-blue-900/20 text-blue-500 rounded-lg">
                  <MessageSquare size={20} />
                </div>
                <div className="space-y-1 flex-1">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-bold text-slate-900 dark:text-slate-100 truncate pr-4">
                      {item.query}
                    </p>
                    <span className="text-[10px] text-slate-400 font-medium shrink-0">
                      {new Date(item.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <p className="text-sm text-slate-500 dark:text-slate-400 line-clamp-2">
                    {item.results.generated_response || "Search results only"}
                  </p>
                </div>
              </div>
            </div>
            
            <div className="mt-4 flex items-center justify-between pt-4 border-t dark:border-slate-800 opacity-0 group-hover:opacity-100 transition-opacity">
              <div className="flex items-center space-x-4">
                <span className="text-xs text-slate-400">
                  {item.results.results?.length || 0} Sources found
                </span>
                <span className="text-xs text-slate-400">•</span>
                <span className="text-xs text-slate-400">
                  {item.results.processing_time?.toFixed(2) || "0.45"}s Latency
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <button className="p-1.5 text-slate-400 hover:text-red-500 transition-colors">
                  <Trash2 size={16} />
                </button>
                <button className="flex items-center text-xs font-bold text-blue-500 uppercase tracking-wider">
                  Re-open <ArrowRight size={14} className="ml-1" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Fixed missing import
import { History } from "lucide-react";
