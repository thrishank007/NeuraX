# NeuraX Design System: Perplexity Search Console

## 1. Visual Philosophy
A search-first, intelligence-grounded workspace modeled after the Perplexity and modern search console experience.
- **Search-First Interaction**: Conversational intelligence starts with an expansive, prominent search capsule with focus mode toggles and local model indicators.
- **Sources-First Grounding**: Every answer is preceded by a horizontal Sources Carousel featuring verified document cards, confidence scores, and instant provenance previews.
- **Slim Iconic Navigation**: Minimal left icon rail with `+ New Thread`, freeing up 100% horizontal canvas space.
- **Interactive Inline Citations**: Markdown responses feature interactive `[1]`, `[2]` citation tags linked to source records.
- **Follow-Up Suggestions**: Every completed answer suggests relevant follow-up query chips.

---

## 2. Typography
- **Primary Interface**: `Inter` (sans-serif, geometric, balanced tracking)
- **Data & Citations**: `JetBrains Mono` (monospace for citations `[1]`, confidence %, file paths, code)

---

## 3. Color Tokens

### Dark Theme (Perplexity Deep Graphite & Ocean Teal)
| Token | Hex | Role |
|---|---|---|
| `--bg` | `#0F1012` | Main canvas background |
| `--surface` | `#181A1D` | Card surfaces, search capsule, side rail |
| `--surface-2` | `#1F2227` | Input wells, active tabs, nested panels |
| `--surface-3` | `#282C34` | Hover states, selected pills |
| `--border` | `rgba(255, 255, 255, 0.08)` | Hairline dividers and card outlines |
| `--border-subtle` | `rgba(255, 255, 255, 0.04)` | Inner borders |
| `--text` | `#F3F4F6` | Primary high-contrast text |
| `--text-muted` | `#9CA3AF` | Secondary labels, descriptions |
| `--text-dim` | `#6B7280` | Subtle metadata, shortcuts, timestamps |
| `--accent` | `#20B2AA` | Perplexity Teal primary accent |
| `--accent-hover` | `#179B94` | Button hover state |
| `--accent-subtle` | `rgba(32, 178, 170, 0.12)` | Active pill background |
| `--signal-cyan` | `#06B6D4` | Embedding & vision tags |
| `--signal-emerald` | `#10B981` | High confidence / ready status |
| `--signal-amber` | `#F59E0B` | Warning / degraded notice |
| `--signal-rose` | `#F43F5E` | Offline / error alert |

### Light Theme (Clean Paper Canvas)
| Token | Hex | Role |
|---|---|---|
| `--bg` | `#F9F9FB` | Off-white canvas background |
| `--surface` | `#FFFFFF` | Elevated cards, search capsule |
| `--surface-2` | `#F1F3F6` | Wells and inputs |
| `--surface-3` | `#E4E7EC` | Hover highlights |
| `--border` | `#E4E7EC` | Card borders |
| `--text` | `#0F172A` | Primary text |
| `--text-muted` | `#475569` | Secondary text |
| `--accent` | `#0D9488` | Teal primary accent |
