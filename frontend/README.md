# Steal-decision frontend

React + TypeScript (Vite) + shadcn/ui web UI for `backend/`'s
steal-decision API. See the repo root `README.md`'s "Web app" section
for the full picture.

## Prerequisites

- Node 18+
- The backend running (see `backend/README.md`) — this app has no
  standalone mode, every prediction and player search goes through it.

## Running

```bash
npm install
npm run dev
```

Vite prints the local URL (usually `http://localhost:5173`, or the
next free port if that's taken). In dev, `/api/*` requests are proxied
to the backend at `http://localhost:8080` (see `vite.config.ts`) — if
the backend is running on a different port, override it:

```bash
BACKEND_URL=http://localhost:9001 npm run dev
```

## Building

```bash
npm run build
```

## Project layout

```
frontend/src/
  lib/types.ts               # mirrors backend/internal/decision's types + API DTOs field-for-field
  lib/api.ts                    # typed fetch wrapper (searchPlayers, predictStealDecision)
  components/
    ui/                           # shadcn/ui primitives (button, select, command, popover, card, badge, ...)
    SituationForm.tsx               # inning/half/outs/base-state/score/target/count/double-steal
    PlayerCombobox.tsx                 # shadcn Command+Popover search, reused for runner/pitcher/catcher
    PlayerStatFields.tsx                 # read-only summary of a selected player's auto-filled stats
    ResultCard.tsx                         # GO/HOLD badge, win probabilities, break-even, sources
```

Player search auto-fills a real player's stats into the situation —
there's no manual stat-override UI by design (see the repo root
README's "Web app" section).

## Adding more shadcn/ui components

```bash
npx shadcn@latest add <component>
```

This project was initialized with `-t vite -b radix -p nova` (Vite
target, Radix-based primitives, "Nova" style preset) — keep new
components consistent with that unless you deliberately want to change
the look.
