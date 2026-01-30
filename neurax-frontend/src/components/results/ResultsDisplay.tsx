"use client";

import { useQueryStore } from "@/lib/store/queryStore";
import { 
  FileText, 
  ExternalLink, 
  ThumbsUp, 
  ThumbsDown, 
  Copy,
  Check
} from "lucide-react";
import { useState } from "react";

export default function ResultsDisplay() {
  const { results, isQuerying } = useQueryStore();
  const [copied, setCopied] = useState(false);

  if (isQuerying) {
    return (
      <div className="w-full max-w-4xl mx-auto space-y-6 mt-12">
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-1/4"></div>
          <div className="h-32 bg-slate-200 dark:bg-slate-800 rounded"></div>
        </div>
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-1/4"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="h-24 bg-slate-200 dark:bg-slate-800 rounded"></div>
            <div className="h-24 bg-slate-200 dark:bg-slate-800 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!results) return null;

  const handleCopy = () => {
    if (results.generated_response) {
      navigator.clipboard.writeText(results.generated_response);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-8 mt-12 pb-12">
      {/* Generated Response */}
      {results.generated_response && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider flex items-center">
              <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
              AI Response
            </h2>
            <div className="flex items-center space-x-2">
              <button 
                onClick={handleCopy}
                className="p-1.5 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 rounded transition-colors"
                title="Copy to clipboard"
              >
                {copied ? <Check size={16} className="text-green-500" /> : <Copy size={16} />}
              </button>
            </div>
          </div>
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 shadow-sm">
            <div className="prose prose-slate dark:prose-invert max-w-none text-slate-800 dark:text-slate-200">
              {results.generated_response.split('\n').map((line, i) => (
                <p key={i} className="mb-4 last:mb-0 leading-relaxed">
                  {line}
                </p>
              ))}
            </div>
            
            <div className="mt-6 flex items-center space-x-4 border-t dark:border-slate-800 pt-4">
              <span className="text-xs text-slate-400">Was this helpful?</span>
              <button className="text-slate-400 hover:text-blue-500 transition-colors">
                <ThumbsUp size={16} />
              </button>
              <button className="text-slate-400 hover:text-red-500 transition-colors">
                <ThumbsDown size={16} />
              </button>
            </div>
          </div>
        </section>
      )}

      {/* Citations / Sources */}
      {results.results && results.results.length > 0 && (
        <section className="space-y-4">
          <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider flex items-center">
            <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
            Sources & Citations
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {results.results.map((source, idx) => (
              <div 
                key={idx} 
                className="group bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4 hover:border-blue-300 dark:hover:border-blue-700 transition-all cursor-pointer shadow-sm"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center">
                    <div className="p-1.5 bg-slate-100 dark:bg-slate-800 rounded text-slate-500 mr-3">
                      <FileText size={16} />
                    </div>
                    <span className="text-xs font-bold text-blue-500 uppercase">
                      Source [{idx + 1}]
                    </span>
                  </div>
                  <span className="text-[10px] font-medium px-1.5 py-0.5 bg-slate-100 dark:bg-slate-800 text-slate-500 rounded">
                    {Math.round(source.score * 100)}% Match
                  </span>
                </div>
                <h3 className="text-sm font-medium text-slate-800 dark:text-slate-200 mb-2 truncate">
                  {source.metadata?.file_path?.split('/').pop() || 'Unknown Document'}
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400 line-clamp-3 leading-relaxed">
                  {source.content_preview || source.content || 'No preview available'}
                </p>
                <div className="mt-3 flex items-center justify-end opacity-0 group-hover:opacity-100 transition-opacity">
                  <button className="text-[10px] flex items-center text-blue-500 font-semibold uppercase tracking-wider">
                    View Document <ExternalLink size={10} className="ml-1" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
