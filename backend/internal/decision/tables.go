package decision

import (
	"context"
	"fmt"

	"basestealmodel/backend/internal/db"
)

// LoadRE24 loads the entire (tiny, 24-row) RE24 table into memory once.
func LoadRE24(ctx context.Context, q db.Querier) (RE24Table, error) {
	rows, err := q.ListRE24Cells(ctx)
	if err != nil {
		return nil, err
	}
	t := make(RE24Table, len(rows))
	for _, r := range rows {
		t[RE24Key{r.BaseCode, int(r.Outs)}] = r.ExpectedRuns
	}
	return t, nil
}

// LoadWinProbTable loads one full table_kind ('after' or 'hold') into
// memory once. See WinProbLookup for why this stays an in-memory map
// rather than per-request SQL: its fallback chain does dynamically-matched
// weighted averages over the table, which is a Go loop, not ad hoc SQL.
func LoadWinProbTable(ctx context.Context, q db.Querier, tableKind string) (WinProbTable, error) {
	rows, err := q.ListWinProbCells(ctx, tableKind)
	if err != nil {
		return nil, err
	}
	t := make(WinProbTable, len(rows))
	for _, r := range rows {
		key := WinProbKey{int(r.InningBucket), int(r.Half), int(r.Outs), r.BaseCode, int(r.ScoreBucket)}
		t[key] = WinProbCell{WinRate: r.WinRate, N: int(r.N)}
	}
	return t, nil
}

// LoadAll builds every in-memory structure a prediction needs -- called
// once at server startup (see cmd/server/main.go), mirroring how
// predict.py's load_tables()/load_model() build the same dicts once at
// Python process startup today.
func LoadAll(ctx context.Context, q db.Querier) (Tables, *Model, Medians, error) {
	re24, err := LoadRE24(ctx, q)
	if err != nil {
		return Tables{}, nil, Medians{}, fmt.Errorf("load re24: %w", err)
	}
	wpAfter, err := LoadWinProbTable(ctx, q, "after")
	if err != nil {
		return Tables{}, nil, Medians{}, fmt.Errorf("load win_prob (after): %w", err)
	}
	wpHold, err := LoadWinProbTable(ctx, q, "hold")
	if err != nil {
		return Tables{}, nil, Medians{}, fmt.Errorf("load win_prob (hold): %w", err)
	}

	coefRows, err := q.ListModelCoefficients(ctx)
	if err != nil {
		return Tables{}, nil, Medians{}, fmt.Errorf("load model coefficients: %w", err)
	}
	meta, err := q.GetModelMeta(ctx)
	if err != nil {
		return Tables{}, nil, Medians{}, fmt.Errorf("load model meta: %w", err)
	}

	model := &Model{Intercept: meta.Intercept}
	for _, c := range coefRows {
		model.FeatureOrder = append(model.FeatureOrder, c.FeatureName)
		model.Coefficients = append(model.Coefficients, c.Coefficient)
	}
	medians := Medians{
		RunnerSprintSpeed: meta.MedianRunnerSprintSpeed,
		RunnerAge:         meta.MedianRunnerAge,
		CatcherPopTime:    meta.MedianCatcherPopTime,
	}

	tables := Tables{RE24: re24, WPTable: wpAfter, WPHold: wpHold}
	return tables, model, medians, nil
}
