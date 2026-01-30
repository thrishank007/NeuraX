import MetricsChart from "@/components/analytics/MetricsChart";
import { 
  Activity, 
  Clock, 
  Cpu, 
  Database, 
  ShieldCheck,
  Zap
} from "lucide-react";

const stats = [
  { name: 'Total Queries', value: '1,284', change: '+12%', icon: Activity, color: 'text-blue-500' },
  { name: 'Avg. Latency', value: '420ms', change: '-5%', icon: Zap, color: 'text-amber-500' },
  { name: 'Documents Indexed', value: '4,592', change: '+231', icon: Database, color: 'text-emerald-500' },
  { name: 'GPU Utilization', value: '68%', change: 'Normal', icon: Cpu, color: 'text-purple-500' },
];

export default function AnalyticsPage() {
  return (
    <div className="space-y-8 pb-12">
      <div>
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white">System Analytics</h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1">
          Monitor performance, usage, and security metrics
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <div key={stat.name} className="bg-white dark:bg-slate-900 p-6 rounded-2xl border dark:border-slate-800 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className={`p-2 rounded-lg bg-slate-50 dark:bg-slate-800 ${stat.color}`}>
                <stat.icon size={20} />
              </div>
              <span className={`text-xs font-bold ${stat.change.startsWith('+') ? 'text-green-500' : stat.change.startsWith('-') ? 'text-blue-500' : 'text-slate-400'}`}>
                {stat.change}
              </span>
            </div>
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400">{stat.name}</p>
            <p className="text-2xl font-bold text-slate-900 dark:text-white mt-1">{stat.value}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border dark:border-slate-800 shadow-sm">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100">Query Volume</h2>
            <select className="text-xs font-medium border-0 focus:ring-0 bg-transparent text-slate-500">
              <option>Last 24 Hours</option>
              <option>Last 7 Days</option>
            </select>
          </div>
          <MetricsChart type="queries" />
        </div>

        <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border dark:border-slate-800 shadow-sm">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100">Latency Trend (ms)</h2>
            <select className="text-xs font-medium border-0 focus:ring-0 bg-transparent text-slate-500">
              <option>Last 24 Hours</option>
              <option>Last 7 Days</option>
            </select>
          </div>
          <MetricsChart type="latency" />
        </div>
      </div>

      <div className="bg-white dark:bg-slate-900 rounded-2xl border dark:border-slate-800 shadow-sm overflow-hidden">
        <div className="p-6 border-b dark:border-slate-800 flex items-center justify-between">
          <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100 flex items-center">
            <ShieldCheck size={20} className="mr-2 text-emerald-500" /> Security Logs
          </h2>
          <button className="text-xs font-bold text-blue-500 uppercase tracking-wider">Export Logs</button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="bg-slate-50 dark:bg-slate-800/50 border-b dark:border-slate-800">
              <tr>
                <th className="px-6 py-3 text-xs font-bold uppercase tracking-wider text-slate-500">Event</th>
                <th className="px-6 py-3 text-xs font-bold uppercase tracking-wider text-slate-500">Status</th>
                <th className="px-6 py-3 text-xs font-bold uppercase tracking-wider text-slate-500">Source</th>
                <th className="px-6 py-3 text-xs font-bold uppercase tracking-wider text-slate-500">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y dark:divide-slate-800">
              {[
                { event: 'Anomalous Query Pattern', status: 'Blocked', source: 'Internal-API', time: '2 mins ago' },
                { event: 'Sensitive Doc Access', status: 'Audit', source: 'User-Admin', time: '14 mins ago' },
                { event: 'Vector Store Backup', status: 'Success', source: 'System-Cron', time: '1 hour ago' },
                { event: 'LLM Model Reload', status: 'Success', source: 'LM-Studio', time: '3 hours ago' },
              ].map((log, i) => (
                <tr key={i} className="hover:bg-slate-50 dark:hover:bg-slate-800/30 transition-colors">
                  <td className="px-6 py-4 text-sm font-medium text-slate-800 dark:text-slate-200">{log.event}</td>
                  <td className="px-6 py-4">
                    <span className={`px-2 py-1 rounded text-[10px] font-bold uppercase ${
                      log.status === 'Blocked' ? 'bg-red-50 text-red-600' : 
                      log.status === 'Success' ? 'bg-green-50 text-green-600' : 
                      'bg-blue-50 text-blue-600'
                    }`}>
                      {log.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-500">{log.source}</td>
                  <td className="px-6 py-4 text-sm text-slate-400">{log.time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
