import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./features/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        "surface-2": "var(--surface-2)",
        "surface-3": "var(--surface-3)",
        border: "var(--border)",
        "border-subtle": "var(--border-subtle)",
        text: "var(--text)",
        muted: "var(--text-muted)",
        dim: "var(--text-dim)",
        accent: "var(--accent)",
        "accent-hover": "var(--accent-hover)",
        "accent-subtle": "var(--accent-subtle)",
        "accent-glow": "var(--accent-glow)",
        "signal-cyan": "var(--signal-cyan)",
        "signal-cyan-subtle": "var(--signal-cyan-subtle)",
        "signal-amber": "var(--signal-amber)",
        "signal-amber-subtle": "var(--signal-amber-subtle)",
        "signal-rose": "var(--signal-rose)",
        "signal-rose-subtle": "var(--signal-rose-subtle)",
        "signal-emerald": "var(--signal-emerald)",
        success: "var(--success)",
        warning: "var(--warning)",
        danger: "var(--danger)",
        info: "var(--info)",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "-apple-system", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      borderRadius: {
        xs: "3px",
        sm: "4px",
        DEFAULT: "6px",
        md: "6px",
        lg: "8px",
        xl: "10px",
        "2xl": "12px",
      },
      boxShadow: {
        "2xs": "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
        xs: "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1)",
        hud: "0 0 0 1px var(--border), 0 4px 12px -2px rgba(0, 0, 0, 0.25)",
      },
      maxWidth: {
        chat: "52rem",
      },
    },
  },
  plugins: [],
};

export default config;
