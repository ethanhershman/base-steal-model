# Steal-decision backend

Go API (chi + sqlc + SQLite) serving the same decision layer as
`src/predict.py` — RE24, win probability, and a logistic-regression
success-probability model — ported to Go so predictions run with no
Python process at request time. See the repo root `README.md`'s "Web
app" section for the full picture.

## Prerequisites

- Go 1.23+ (developed against 1.26)
- `backend/data/app.db` — already committed, so this runs out of the box.
  Regenerate it after retraining the model or refreshing the Retrosheet
  data with (from the repo root):
  ```bash
  python -m src.export_web_data
  ```

## Running

```bash
go run ./cmd/server
```

Listens on `:8080` by default. Logs how many RE24/win-probability
cells and model features it loaded — a quick sanity check that
`app.db` isn't stale or corrupt.

### Environment variables

| Variable | Default | Notes |
|---|---|---|
| `DB_PATH` | `data/app.db` | Path to the SQLite artifact (relative to `backend/` unless absolute). |
| `PORT` | `8080` | Override if something else on the machine holds 8080. |
| `CORS_ORIGIN` | `http://localhost:5173` | Must match wherever the frontend is actually running. |

Example, if 8080 is taken locally:
```bash
PORT=9001 go run ./cmd/server
```
(and set `BACKEND_URL=http://localhost:9001` when starting the
frontend — see `frontend/README.md`.)

## API

- `GET /api/health` → `{"status":"ok"}`
- `GET /api/players/search?role={runner|pitcher|catcher}&q={prefix}&limit=10` — real players by name, with their stats embedded (no second lookup needed).
- `POST /api/predict` — a game situation in, a GO/HOLD recommendation out. Mirrors `predict_steal_decision`'s kwargs/response exactly (see `internal/api/dto.go`).

```bash
curl -s -X POST localhost:8080/api/predict -H "Content-Type: application/json" -d '{
  "inning": 9, "half": 1, "outs": 2, "base_code": "1__", "score_diff": -1, "target": "2",
  "runner_sprint_speed": 29.5, "catcher_pop_time": 1.95,
  "runner_prior_sr": 0.82, "runner_prior_att": 25
}'
```

## Project layout

```
backend/
  sql/schema.sql          # single source of truth for the DB shape (sqlc + src/export_web_data.py both read it)
  sql/queries/*.sql        # sqlc query annotations
  internal/db/              # sqlc-generated (DO NOT EDIT) -- regenerate with `sqlc generate`
  internal/decision/          # the ported decision layer -- pure Go, no DB/HTTP deps
  internal/api/                 # chi handlers + request/response DTOs
  internal/config/                # env var loading
  cmd/server/                       # wires it all together, starts the HTTP server
```

`internal/decision` loads RE24/win-probability/model data into memory
once at startup (see `tables.go`) rather than querying per request —
see that package's doc comments for why. Player search stays live SQL
via sqlc, since `LIKE`-based search is a genuinely good fit for a real
query engine.

## Testing

```bash
go test ./...
```

Includes `internal/decision/golden_test.go`, which checks the Go port
against 47 fixtures generated directly from `predict_steal_decision`
(`python -m src.export_golden_fixtures` regenerates
`internal/decision/testdata/golden_fixtures.json` — do this after
changing any ported decision logic, and re-verify the Go tests still
pass).

If you change `sql/schema.sql` or add a query in `sql/queries/`, run:
```bash
sqlc generate
```
