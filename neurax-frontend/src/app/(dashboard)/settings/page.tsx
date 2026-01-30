"use client";

import { useState } from "react";
import { 
  Settings, 
  Link2, 
  Cpu, 
  Shield, 
  Moon, 
  Database,
  Check
} from "lucide-react";

export default function SettingsPage() {
  const [saveStatus, setSaveStatus] = useState(false);

  const handleSave = () => {
    setSaveStatus(true);
    setTimeout(() => setSaveStatus(false), 2000);
  };

  return (
    <div className="max-w-4xl space-y-8 pb-12">
      <div>
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white flex items-center">
          <Settings className="mr-3 text-slate-400" /> Settings
        </h1>
        <p className="text-slate-500 dark:text-slate-400 mt-1">
          Configure system parameters and preferences
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="md:col-span-2 space-y-6">
          {/* LM Studio Connection */}
          <section className="bg-white dark:bg-slate-900 border dark:border-slate-800 rounded-2xl p-6 shadow-sm">
            <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-6 flex items-center">
              <Link2 size={16} className="mr-2" /> LLM Configuration
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  LM Studio Base URL
                </label>
                <input 
                  type="text" 
                  defaultValue="http://localhost:1234"
                  className="w-full bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Default Reasoning Model
                </label>
                <select className="w-full bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                  <option>Qwen3 4B (Reasoning)</option>
                  <option>Gemma 3n (Multimodal)</option>
                  <option>Mistral 7B</option>
                </select>
              </div>
            </div>
          </section>

          {/* Performance Tuning */}
          <section className="bg-white dark:bg-slate-900 border dark:border-slate-800 rounded-2xl p-6 shadow-sm">
            <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-6 flex items-center">
              <Cpu size={16} className="mr-2" /> Performance Tuning
            </h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-700 dark:text-slate-200">GPU Acceleration</p>
                  <p className="text-xs text-slate-500">Enable CUDA support for embeddings</p>
                </div>
                <div className="relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent bg-blue-600 transition-colors duration-200 ease-in-out">
                  <span className="translate-x-5 inline-block h-5 w-5 transform rounded-full bg-white transition duration-200 ease-in-out" />
                </div>
              </div>
              <div className="pt-4 border-t dark:border-slate-800">
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Batch Processing Size
                </label>
                <input 
                  type="range" 
                  min="1" 
                  max="32" 
                  defaultValue="8"
                  className="w-full h-2 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                  <span>1 doc</span>
                  <span>32 docs</span>
                </div>
              </div>
            </div>
          </section>

          {/* Security Policy */}
          <section className="bg-white dark:bg-slate-900 border dark:border-slate-800 rounded-2xl p-6 shadow-sm">
            <h2 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-6 flex items-center">
              <Shield size={16} className="mr-2" /> Security & Privacy
            </h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-700 dark:text-slate-200">Strict Offline Mode</p>
                  <p className="text-xs text-slate-500">Disable all external network calls</p>
                </div>
                <div className="relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent bg-blue-600 transition-colors duration-200 ease-in-out">
                  <span className="translate-x-5 inline-block h-5 w-5 transform rounded-full bg-white transition duration-200 ease-in-out" />
                </div>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-700 dark:text-slate-200">Audit Logging</p>
                  <p className="text-xs text-slate-500">Log all interactions and model queries</p>
                </div>
                <div className="relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent bg-blue-600 transition-colors duration-200 ease-in-out">
                  <span className="translate-x-5 inline-block h-5 w-5 transform rounded-full bg-white transition duration-200 ease-in-out" />
                </div>
              </div>
            </div>
          </section>

          <div className="flex justify-end pt-4">
            <button 
              onClick={handleSave}
              className="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-bold transition-all shadow-lg shadow-blue-500/20 flex items-center"
            >
              {saveStatus ? <><Check size={18} className="mr-2" /> Settings Saved</> : "Save Changes"}
            </button>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-blue-600 rounded-2xl p-6 text-white shadow-lg shadow-blue-500/20">
            <h3 className="text-lg font-bold mb-2">System Status</h3>
            <p className="text-sm text-blue-100 mb-6">Current hardware utilization and health.</p>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1 font-medium">
                  <span>CPU Usage</span>
                  <span>24%</span>
                </div>
                <div className="w-full bg-blue-400/30 rounded-full h-1.5">
                  <div className="bg-white rounded-full h-1.5 w-[24%]" />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1 font-medium">
                  <span>RAM Usage</span>
                  <span>12.4 GB</span>
                </div>
                <div className="w-full bg-blue-400/30 rounded-full h-1.5">
                  <div className="bg-white rounded-full h-1.5 w-[45%]" />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1 font-medium">
                  <span>Vector Index</span>
                  <span>Healthy</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-slate-900 border dark:border-slate-800 rounded-2xl p-6 shadow-sm">
            <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100 mb-4">Storage Info</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-500">Indexed Docs</span>
                <span className="text-xs font-bold text-slate-800 dark:text-slate-100">4,592</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-500">Vector Embeddings</span>
                <span className="text-xs font-bold text-slate-800 dark:text-slate-100">1.2 GB</span>
              </div>
              <div className="flex items-center justify-between pt-3 border-t dark:border-slate-800">
                <button className="text-xs font-bold text-red-500 uppercase tracking-wider">Clear Vector Cache</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
