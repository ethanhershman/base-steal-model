package decision

import "math"

func inningBucket(inning int) int {
	if inning > MaxInning {
		return MaxInning
	}
	return inning
}

func scoreBucket(scoreDiff int) int {
	if scoreDiff > ScoreClip {
		return ScoreClip
	}
	if scoreDiff < -ScoreClip {
		return -ScoreClip
	}
	return scoreDiff
}

// WinProbLookup ports win_probability.win_prob_lookup's 5-stage fallback
// chain as a literal ordered sequence -- NOT a "smarter" generalization.
// Every stage after (1) holds inning bucket AND half fixed and only ever
// relaxes the score window/base code: that's the entire reason this table
// exists (see the module docstring on walk-off dynamics), so a fallback
// that crossed an inning/half boundary would wash out exactly the signal
// being asked for.
//
// Returns (winProb, n, source) where n is +Inf for the hardcoded certainty
// case (see MinN's JSON handling in types.go) and source names which
// fallback stage produced the answer.
func WinProbLookup(table WinProbTable, inning, half, outs int, baseCode string, scoreDiff int, minN int) (float64, float64, string) {
	ib, sb := inningBucket(inning), scoreBucket(scoreDiff)

	// Stage 1: logical certainty, not an empirical estimate. Tested on the
	// RAW score_diff (not the clamped bucket) to match the Python source
	// exactly, though in practice sign is preserved by clamping so this
	// never actually diverges from testing sb < 0.
	if outs == 3 && half == 1 && ib >= MaxInning && scoreDiff < 0 {
		return 0.0, math.Inf(1), "certain (home team trailing, game over)"
	}

	// Stage 2: exact cell.
	if cell, ok := table[WinProbKey{ib, half, outs, baseCode, sb}]; ok && cell.N >= minN {
		return cell.WinRate, float64(cell.N), "exact"
	}

	// Stage 3: widen the score-margin window (+-1 bucket), same inning/half/outs/base.
	if rate, n, ok := weightedAverage(table, func(k WinProbKey) bool {
		return k.InningBucket == ib && k.Half == half && k.Outs == outs &&
			k.BaseCode == baseCode && absInt(k.ScoreBucket-sb) <= 1
	}, minN); ok {
		return rate, n, "widened score window"
	}

	// Stage 4: same inning/half/outs, widened score window, averaged over base codes.
	if rate, n, ok := weightedAverage(table, func(k WinProbKey) bool {
		return k.InningBucket == ib && k.Half == half && k.Outs == outs &&
			absInt(k.ScoreBucket-sb) <= 1
	}, minN); ok {
		return rate, n, "widened score window, all base codes"
	}

	// Stage 5: last resort, same inning/half/outs, ANY score margin -- still
	// never crosses into a different inning bucket.
	w, n := 0.0, 0
	for k, cell := range table {
		if k.InningBucket == ib && k.Half == half && k.Outs == outs {
			w += cell.WinRate * float64(cell.N)
			n += cell.N
		}
	}
	if n > 0 {
		return w / float64(n), float64(n), "any score margin, all base codes (low confidence)"
	}

	return 0.5, 0, "no data (default)"
}

func weightedAverage(table WinProbTable, match func(WinProbKey) bool, minN int) (rate, n float64, ok bool) {
	w, count := 0.0, 0
	for k, cell := range table {
		if match(k) {
			w += cell.WinRate * float64(cell.N)
			count += cell.N
		}
	}
	if count >= minN {
		return w / float64(count), float64(count), true
	}
	return 0, 0, false
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// WinProbBreakEven ports win_probability.win_prob_break_even. Mirrors
// BreakEvenRate's reward/cost structure, but the currency is P(win)
// instead of expected runs. `cur` is looked up in holdTable (the
// "before the decision" baseline); wpSucc/wpCaught in table (the
// after-success/after-caught table) -- two distinct tables, never merged.
func WinProbBreakEven(table, holdTable WinProbTable, inning, half, outs int, baseCode string, scoreDiff int, target string, minN int) (breakEven, reward, cost, minNSeen float64, sources []string) {
	cur, nCur, srcCur := WinProbLookup(holdTable, inning, half, outs, baseCode, scoreDiff, minN)

	succBase, runBonus := StateAfterSuccess(baseCode, target)
	succScore := scoreDiff + runBonus
	wpSucc, nSucc, srcSucc := WinProbLookup(table, inning, half, outs, succBase, succScore, minN)

	var wpCaught, nCaught float64
	var srcCaught string
	if outs >= 2 {
		wpCaught, nCaught, srcCaught = WinProbLookup(table, inning, half, 3, "END", scoreDiff, minN)
	} else {
		caughtBase := StateAfterCaught(baseCode, target)
		wpCaught, nCaught, srcCaught = WinProbLookup(table, inning, half, outs+1, caughtBase, scoreDiff, minN)
	}

	reward = wpSucc - cur
	cost = cur - wpCaught
	denom := reward + cost
	// Reward/cost SHOULD both be >= 0, but small-sample cells can disagree
	// by noise alone -- clip to a valid probability rather than reporting a
	// break-even outside [0,1] (see win_probability.py's own comment on this).
	if denom > 0 {
		be := cost / denom
		if be < 0 {
			be = 0
		}
		if be > 1 {
			be = 1
		}
		breakEven = be
	} else {
		breakEven = 1.0
	}

	minNSeen = math.Min(nCur, math.Min(nSucc, nCaught))
	sources = []string{srcCur, srcSucc, srcCaught}
	return breakEven, reward, cost, minNSeen, sources
}

// IsHighLeverage ports win_probability.is_high_leverage.
func IsHighLeverage(inning, scoreDiff, leverageInnings, leverageMargin int) bool {
	return inning >= leverageInnings && absInt(scoreDiff) <= leverageMargin
}
