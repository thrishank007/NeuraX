import { ShieldAlert, Lock, Eye, AlertTriangle, ShieldCheck } from "lucide-react";

export default function SecurityPage() {
  return (
    <div className="max-w-4xl space-y-8 pb-12">
      <div>
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white flex items-center">
          <ShieldAlert className="mr-3 text-red-500" /> Security Control
        </h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1">
          Monitor system integrity and access logs
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-slate-900 border dark:border-slate-800 rounded-2xl p-6 shadow-sm">
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2 bg-red-50 dark:bg-red-900/20 text-red-500 rounded-lg">
              <AlertTriangle size={20} />
            </div>
            <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100">Anomaly Detection</h2>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-red-50/50 dark:bg-red-900/10 border border-red-100 dark:border-red-900/30 rounded-xl">
              <div className="flex items-center space-x-3">
                <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
                <span className="text-sm font-medium text-red-900 dark:text-red-300">Potential data exfiltration detected</span>
              </div>
              <button className="text-[10px] font-bold text-red-600 uppercase">View</button>
            </div>
            <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded-xl">
              <div className="flex items-center space-x-3 text-slate-600 dark:text-slate-400">
                <span className="w-2 h-2 bg-slate-300 dark:bg-slate-600 rounded-full"></span>
                <span className="text-sm font-medium">Model query pattern normalized</span>
              </div>
              <span className="text-[10px] font-bold text-slate-400 uppercase">Resolved</span>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-900 border dark:border-slate-800 rounded-2xl p-6 shadow-sm">
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2 bg-green-50 dark:bg-green-900/20 text-green-500 rounded-lg">
              <ShieldCheck size={20} />
            </div>
            <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100">System Integrity</h2>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-600 dark:text-slate-400">Vector Store Encryption</span>
              <span className="text-xs font-bold text-green-500 flex items-center">
                <Lock size={12} className="mr-1" /> AES-256
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-600 dark:text-slate-400">Model Verification</span>
              <span className="text-xs font-bold text-green-500 flex items-center">
                <Check size={12} className="mr-1" /> Verified
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-600 dark:text-slate-400">Data Privacy Level</span>
              <span className="text-xs font-bold text-blue-500 flex items-center">
                <Eye size={12} className="mr-1" /> Air-gapped
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 border dark:border-slate-800 rounded-2xl shadow-sm overflow-hidden">
        <div className="p-6 border-b dark:border-slate-800">
          <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100">Audit Logs</h2>
        </div>
        <div className="divide-y dark:divide-slate-800">
          {[
            { action: 'Database Indexing', user: 'System', detail: 'Completed in 12s', time: '10 mins ago' },
            { action: 'Sensitive Document Access', user: 'Admin', detail: 'viewed confidential_report.pdf', time: '1 hour ago' },
            { action: 'Model Selection Changed', user: 'Admin', detail: 'switched to Qwen3 4B', time: '4 hours ago' },
            { action: 'Batch Upload', user: 'User-A', detail: '24 files processed', time: 'Yesterday' },
          ].map((log, i) => (
            <div key={i} className="px-6 py-4 flex items-center justify-between hover:bg-slate-50 dark:hover:bg-slate-800/30 transition-colors">
              <div>
                <p className="text-sm font-bold text-slate-900 dark:text-slate-100">{log.action}</p>
                <p className="text-xs text-slate-500 mt-0.5">{log.detail}</p>
              </div>
              <div className="text-right">
                <p className="text-xs font-medium text-slate-700 dark:text-slate-300">{log.user}</p>
                <p className="text-[10px] text-slate-400">{log.time}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

import { Check } from "lucide-react";
