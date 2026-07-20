package decision

import (
	"math"
	"testing"
)

// A small, hand-constructed table so each of the 5 fallback stages
// triggers deterministically -- not hoping the real ~8,100-row table
// happens to exercise all of them. minN=5 throughout (passed explicitly
// to WinProbLookup, not tied to the real MinCellN=20 constant) to keep the
// fixture small and readable.
func fixtureWinProbTable() WinProbTable {
	return WinProbTable{
		// Stage 2 (exact): plenty of samples right at the query cell.
		{5, 0, 1, "1__", 0}: {WinRate: 0.60, N: 10},

		// Stage 3 (widened score window, same base code): exact cell at
		// sb=0 alone has n=2 (<5), but sb in {-1,0,1} sums to 6 (>=5).
		{5, 0, 1, "_2_", 0}:  {WinRate: 0.50, N: 2},
		{5, 0, 1, "_2_", 1}:  {WinRate: 0.70, N: 2},
		{5, 0, 1, "_2_", -1}: {WinRate: 0.40, N: 2},

		// Stage 4 (widened score window, all base codes): "__3" alone
		// (any score window) only ever has n=1 here, but summing every
		// base code within sb in {-1,0,1} reaches 5.
		{6, 0, 2, "__3", 0}: {WinRate: 0.30, N: 1},
		{6, 0, 2, "1__", 0}: {WinRate: 0.35, N: 2},
		{6, 0, 2, "_2_", 1}: {WinRate: 0.40, N: 2},

		// Stage 5 (any score margin, all base codes): only data far
		// outside the +-1 score window, so stages 3/4 both come up empty.
		{7, 1, 0, "1__", 4}: {WinRate: 0.90, N: 6},

		// Decoy at a DIFFERENT inning/half, to prove the "never cross
		// inning/half" invariant -- if a fallback stage ever picked this
		// up, the stage-5 test below would see a blended, wrong result.
		{8, 1, 0, "1__", 0}: {WinRate: 0.10, N: 100},
	}
}

func TestWinProbLookupStage2Exact(t *testing.T) {
	rate, n, src := WinProbLookup(fixtureWinProbTable(), 5, 0, 1, "1__", 0, 5)
	if src != "exact" {
		t.Fatalf("source = %q, want %q", src, "exact")
	}
	if rate != 0.60 || n != 10 {
		t.Fatalf("got (%v, %v), want (0.60, 10)", rate, n)
	}
}

func TestWinProbLookupStage3WidenedScoreWindow(t *testing.T) {
	rate, n, src := WinProbLookup(fixtureWinProbTable(), 5, 0, 1, "_2_", 0, 5)
	if src != "widened score window" {
		t.Fatalf("source = %q, want %q", src, "widened score window")
	}
	wantRate := (0.50*2 + 0.70*2 + 0.40*2) / 6
	if math.Abs(rate-wantRate) > 1e-9 || n != 6 {
		t.Fatalf("got (%v, %v), want (%v, 6)", rate, n, wantRate)
	}
}

func TestWinProbLookupStage4WidenedAllBaseCodes(t *testing.T) {
	rate, n, src := WinProbLookup(fixtureWinProbTable(), 6, 0, 2, "__3", 0, 5)
	if src != "widened score window, all base codes" {
		t.Fatalf("source = %q, want %q", src, "widened score window, all base codes")
	}
	wantRate := (0.30*1 + 0.35*2 + 0.40*2) / 5
	if math.Abs(rate-wantRate) > 1e-9 || n != 5 {
		t.Fatalf("got (%v, %v), want (%v, 5)", rate, n, wantRate)
	}
}

func TestWinProbLookupStage5AnyScoreMargin(t *testing.T) {
	rate, n, src := WinProbLookup(fixtureWinProbTable(), 7, 1, 0, "1__", 0, 5)
	if src != "any score margin, all base codes (low confidence)" {
		t.Fatalf("source = %q, want %q", src, "any score margin, all base codes (low confidence)")
	}
	if rate != 0.90 || n != 6 {
		t.Fatalf("got (%v, %v), want (0.90, 6) -- decoy at a different inning must not leak in", rate, n)
	}
}

func TestWinProbLookupStage6NoData(t *testing.T) {
	rate, n, src := WinProbLookup(fixtureWinProbTable(), 8, 0, 1, "1__", 0, 5)
	if src != "no data (default)" {
		t.Fatalf("source = %q, want %q", src, "no data (default)")
	}
	if rate != 0.5 || n != 0 {
		t.Fatalf("got (%v, %v), want (0.5, 0)", rate, n)
	}
}

func TestWinProbLookupCertainCaseShortCircuitsRegardlessOfTable(t *testing.T) {
	// Home team trailing, half-inning just ended, 9th or later: a logical
	// certainty, not an empirical estimate -- fires even on an empty table.
	rate, n, src := WinProbLookup(WinProbTable{}, 9, 1, 3, "END", -1, 20)
	if rate != 0.0 || !math.IsInf(n, 1) {
		t.Fatalf("got (%v, %v), want (0.0, +Inf)", rate, n)
	}
	if src != "certain (home team trailing, game over)" {
		t.Fatalf("source = %q", src)
	}
}

func TestWinProbLookupCertainCaseDoesNotFireWhenLeading(t *testing.T) {
	_, _, src := WinProbLookup(WinProbTable{}, 9, 1, 3, "END", 1, 20)
	if src == "certain (home team trailing, game over)" {
		t.Fatalf("certain case should not fire when the home team is NOT trailing")
	}
}

func TestIsHighLeverageInningBoundary(t *testing.T) {
	if IsHighLeverage(6, 0, 7, 3) {
		t.Fatal("inning 6 should not be high leverage (boundary is 7)")
	}
	if !IsHighLeverage(7, 0, 7, 3) {
		t.Fatal("inning 7 should be high leverage")
	}
}

func TestIsHighLeverageScoreMarginBoundary(t *testing.T) {
	if !IsHighLeverage(8, 3, 7, 3) {
		t.Fatal("score margin 3 should be high leverage (boundary is <=3)")
	}
	if IsHighLeverage(8, 4, 7, 3) {
		t.Fatal("score margin 4 should not be high leverage")
	}
	if !IsHighLeverage(8, -3, 7, 3) {
		t.Fatal("score margin -3 should be high leverage (sign shouldn't matter)")
	}
}
