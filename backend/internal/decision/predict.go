package decision

import "math"

// PredictStealDecision ports predict.py's predict_steal_decision exactly --
// the main orchestration the API's /api/predict handler calls.
func PredictStealDecision(tables Tables, model *Model, medians Medians, s Situation) Decision {
	row := BuildFeatureRow(medians, s)
	pSuccess := model.Predict(row)

	// Always compute win probability for display, regardless of which
	// layer ends up making the decision -- see predict.py's module
	// docstring: early in the game win probability is noisier and RE24 is
	// the right optimization target, but it's still informative to show.
	succBase, runBonus := StateAfterSuccess(s.BaseCode, s.Target)
	succScore := s.ScoreDiff + runBonus
	wpCurrent, nCur, srcCur := WinProbLookup(tables.WPHold, s.Inning, s.Half, s.Outs, s.BaseCode, s.ScoreDiff, MinCellN)
	wpIfSuccess, nSucc, srcSucc := WinProbLookup(tables.WPTable, s.Inning, s.Half, s.Outs, succBase, succScore, MinCellN)

	highLeverage := IsHighLeverage(s.Inning, s.ScoreDiff, 7, 3)

	// Defaults match the RE24 path's shape (2-element sources, min of the
	// two win-prob lookups already done above) -- only overridden below if
	// the win-probability path decides instead. Sources is genuinely
	// variable-length: 2 elements here, 3 from WinProbBreakEven.
	layer := "WP"
	minN := math.Min(nCur, nSucc)
	sources := []string{srcCur, srcSucc}

	_, hasRE24 := tables.RE24[RE24Key{s.BaseCode, s.Outs}]

	var breakEven float64
	if !highLeverage && hasRE24 {
		breakEven, _, _ = BreakEvenRate(tables.RE24, s.BaseCode, s.Outs, s.Target)
		layer = "RE24"
	} else {
		// Covers two distinct real-world cases -- genuine high leverage, OR
		// a genuine RE24 coverage gap for an exotic base/out combo -- with
		// a different layer label for each, purely for UI transparency;
		// behaviorally identical either way, and never panics on a combo
		// RE24 has no data for.
		var minNSeen float64
		var srcs []string
		breakEven, _, _, minNSeen, srcs = WinProbBreakEven(
			tables.WPTable, tables.WPHold, s.Inning, s.Half, s.Outs, s.BaseCode, s.ScoreDiff, s.Target, MinCellN)
		minN = minNSeen
		sources = srcs
		if !highLeverage {
			layer = "WP (RE24 had no data)"
		}
	}

	decisionLabel := "HOLD"
	if pSuccess > breakEven { // strict > -- an exact tie is HOLD
		decisionLabel = "GO"
	}

	return Decision{
		Inning: s.Inning, Half: s.Half, Outs: s.Outs, BaseCode: s.BaseCode,
		ScoreDiff: s.ScoreDiff, Target: s.Target,
		WinProbCurrent:   wpCurrent,
		WinProbIfSuccess: wpIfSuccess,
		BreakEven:        breakEven,
		PSuccess:         pSuccess,
		DecisionLabel:    decisionLabel,
		Layer:            layer,
		MinN:             MinN(minN),
		LowConfidence:    minN < MinCellN,
		Sources:          sources,
	}
}
