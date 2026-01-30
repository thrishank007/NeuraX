"use client";

import { Bell, Moon, Sun, User } from "lucide-react";
import { useState, useEffect } from "react";

export default function Header() {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const isDark = document.documentElement.classList.contains('dark');
      setIsDarkMode(isDark);
    }
  }, []);

  const toggleDarkMode = () => {
    const newMode = !isDarkMode;
    setIsDarkMode(newMode);
    if (newMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  return (
    <header className="h-16 border-b bg-white dark:bg-slate-950 px-6 flex items-center justify-between transition-colors">
      <div className="flex items-center">
        <h1 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
          Secure multimodal document intelligence
        </h1>
      </div>
      <div className="flex items-center space-x-4">
        <button 
          onClick={toggleDarkMode}
          className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500 dark:text-slate-400"
        >
          {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
        </button>
        <button className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500 dark:text-slate-400">
          <Bell size={20} />
        </button>
        <div className="h-8 w-px bg-slate-200 dark:bg-slate-800 mx-2" />
        <button className="flex items-center space-x-2 p-1 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800">
          <div className="h-8 w-8 rounded-full bg-slate-200 dark:bg-slate-800 flex items-center justify-center">
            <User size={20} className="text-slate-500" />
          </div>
          <span className="text-sm font-medium text-slate-700 dark:text-slate-200 hidden md:inline">Admin</span>
        </button>
      </div>
    </header>
  );
}
