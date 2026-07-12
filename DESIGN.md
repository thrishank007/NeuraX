# NeuraX Design System

## Direction

**Calm technical workspace for local document intelligence.**  
Feels like a precision instrument: restrained color, strong typography hierarchy, visible status, source panels that reward inspection.

**Not:** generic AI purple gradients, glassmorphism, oversized heroes, ChatGPT clone skins, decorative motion.

## Typography

| Role | Stack | Notes |
|---|---|---|
| UI / body | `IBM Plex Sans`, system-ui, sans-serif | Readable at dense sizes |
| Code / paths / IDs | `IBM Plex Mono`, ui-monospace | Filenames, scores, endpoints |
| Scale | 12 / 14 / 16 / 18 / 24 / 32 | Default body 14px; chat text 15–16px |

## Spacing scale

`4, 8, 12, 16, 24, 32, 48` — prefer 8px grid. Dense panels use 8–12; page gutters 16–24.

## Color tokens

### Light

| Token | Value | Use |
|---|---|---|
| `--bg` | `#F4F1EA` | App background (warm paper) |
| `--surface` | `#FFFcf7` | Panels |
| `--surface-2` | `#EBE6DC` | Nested wells |
| `--border` | `#D4CDBF` | Dividers |
| `--text` | `#1C1915` | Primary text |
| `--text-muted` | `#5C564C` | Secondary |
| `--accent` | `#0F6E56` | Actions, focus (deep teal) |
| `--accent-hover` | `#0B5A46` | Hover |
| `--accent-subtle` | `#D8EDE6` | Selected rows |

### Dark

| Token | Value | Use |
|---|---|---|
| `--bg` | `#121410` | App background |
| `--surface` | `#1A1D18` | Panels |
| `--surface-2` | `#242820` | Nested |
| `--border` | `#33382E` | Dividers |
| `--text` | `#ECE7DC` | Primary |
| `--text-muted` | `#A39E91` | Secondary |
| `--accent` | `#3D9B7A` | Actions |
| `--accent-hover` | `#4FB08C` | Hover |
| `--accent-subtle` | `#1E332B` | Selected |

### Status (light / dark pair — always pair icon + text)

| Status | Light | Dark |
|---|---|---|
| Success / ready | `#0F6E56` | `#3D9B7A` |
| Warning | `#9A6700` | `#D4A017` |
| Error | `#B42318` | `#F97066` |
| Info / running | `#175CD3` | `#84ADFF` |
| Neutral offline | `#5C564C` | `#A39E91` |

**Contrast:** body text ≥ 4.5:1 on surfaces; muted text still ≥ 4.5:1 where used for essential labels.

## Layout

- App shell: collapsible left nav (~220px / 56px collapsed), main work area, optional right source drawer (~320–380px)  
- Max content width for chat column ~720–800px; sources expand beside it on large screens  
- Avoid card-in-card nesting; use borders and surface shifts  

## Components

- **Buttons:** solid accent primary; ghost secondary; destructive uses error color text/border  
- **Inputs:** 1px border, 6px radius (restrained—not pill), visible focus ring `2px accent`  
- **Tables / lists:** compact rows, mono for scores  
- **Status chip:** icon + label; never color-only  
- **Dialogs:** modal for delete confirm; Esc closes; focus trap  

## Interaction states

Hover, focus-visible, active, disabled (40% opacity + no pointer), loading (inline spinner + text).

## Motion

- Duration 120–180ms ease-out for panels  
- Respect `prefers-reduced-motion: reduce` — disable non-essential transitions  
- No continuous decorative animation  

## Accessibility

- Semantic landmarks (`nav`, `main`, `aside`)  
- Skip link to main content  
- Keyboard: Tab order logical; chat composer Ctrl/Cmd+Enter submit  
- ARIA live regions for status and streaming progress  
- Hit targets ≥ 40px where practical  

## Responsive

| Breakpoint | Behavior |
|---|---|
| < 768px | Nav as bottom or drawer; source panel full-screen sheet |
| 768–1024 | Collapsed nav; stack source under chat |
| > 1024 | Three-zone workspace |
| > 1440 | Comfortable gutters; no stretched sparse hero |

## Anti-patterns

- Purple/pink AI gradients  
- Excessive glass blur  
- Huge rounded marketing cards  
- Random icon packs without meaning  
- Hiding LM Studio / backend failures  
