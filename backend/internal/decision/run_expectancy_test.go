package decision

import "testing"

// realRE24 is the actual committed RE24 table (data/sample/re24_2023_2025.csv)
// as of this writing -- using real values rather than made-up ones so these
// tests double as a sanity check against the real data, not just internally
// self-consistent numbers.
func realRE24() RE24Table {
	return RE24Table{
		{"___", 0}: 0.500, {"___", 1}: 0.267, {"___", 2}: 0.102,
		{"1__", 0}: 0.894, {"1__", 1}: 0.528, {"1__", 2}: 0.231,
		{"_2_", 0}: 1.124, {"_2_", 1}: 0.680, {"_2_", 2}: 0.323,
		{"__3", 0}: 1.390, {"__3", 1}: 0.944, {"__3", 2}: 0.353,
		{"12_", 0}: 1.504, {"12_", 1}: 0.939, {"12_", 2}: 0.458,
		{"1_3", 0}: 1.841, {"1_3", 1}: 1.202, {"1_3", 2}: 0.503,
		{"_23", 0}: 1.930, {"_23", 1}: 1.364, {"_23", 2}: 0.558,
		{"123", 0}: 2.320, {"123", 1}: 1.587, {"123", 2}: 0.774,
	}
}

func TestStateAfterSuccessNormalSingleRunner(t *testing.T) {
	state, runs := StateAfterSuccess("1__", "2")
	if state != "_2_" || runs != 0 {
		t.Fatalf("got (%q, %d), want (\"_2_\", 0)", state, runs)
	}
}

func TestStateAfterSuccessStealOfHomeScores(t *testing.T) {
	state, runs := StateAfterSuccess("__3", "H")
	if state != "___" || runs != 1 {
		t.Fatalf("got (%q, %d), want (\"___\", 1)", state, runs)
	}
}

func TestStateAfterSuccessDoubleStealCascadesTheOccupiedRunner(t *testing.T) {
	// Runners on 1st and 2nd; the trailing runner steals 2nd, so the
	// runner already on 2nd must simultaneously be advancing to 3rd (two
	// runners can't occupy the same base) -- not overwritten/lost.
	state, runs := StateAfterSuccess("12_", "2")
	if state != "_23" || runs != 0 {
		t.Fatalf("got (%q, %d), want (\"_23\", 0)", state, runs)
	}
}

func TestStateAfterSuccessDoubleStealScoresWhenCascadeReachesHome(t *testing.T) {
	// Runners on 2nd and 3rd; the trailing runner steals 3rd, so the
	// runner already on 3rd is pushed home and scores.
	state, runs := StateAfterSuccess("_23", "3")
	if state != "__3" || runs != 1 {
		t.Fatalf("got (%q, %d), want (\"__3\", 1)", state, runs)
	}
}

func TestStateAfterSuccessTripleStealCascadesThroughEveryBase(t *testing.T) {
	// Bases loaded, runner on 1st steals 2nd: pushes 1st's occupant to
	// 2nd (already occupied) -> 2nd's occupant to 3rd (already occupied)
	// -> 3rd's occupant scores.
	state, runs := StateAfterSuccess("123", "2")
	if state != "_23" || runs != 1 {
		t.Fatalf("got (%q, %d), want (\"_23\", 1)", state, runs)
	}
}

func TestCaughtStealingThirdOutEndsInning(t *testing.T) {
	re24 := realRE24()
	// With 2 outs, a caught stealing ends the inning (RE=0 after), so the
	// cost of getting caught should equal the full current-state RE.
	_, _, cost := BreakEvenRate(re24, "1__", 2, "2")
	want := re24[RE24Key{"1__", 2}]
	if diff := cost - want; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("cost = %v, want %v", cost, want)
	}
}

func TestBreakEvenReasonableForSteal2nd(t *testing.T) {
	re24 := realRE24()
	for _, outs := range []int{0, 1, 2} {
		be, reward, cost := BreakEvenRate(re24, "1__", outs, "2")
		if be < 0.60 || be > 0.85 {
			t.Errorf("outs=%d: break-even %.3f outside plausible range [0.60, 0.85]", outs, be)
		}
		if reward <= 0 {
			t.Errorf("outs=%d: reward = %v, want > 0", outs, reward)
		}
		if cost <= 0 {
			t.Errorf("outs=%d: cost = %v, want > 0", outs, cost)
		}
	}
}

func TestDoubleStealBreakEvenIsSaneNotCorrupted(t *testing.T) {
	// Before the cascade fix (see the Python history: src/run_expectancy.py),
	// stealing into an occupied base produced a negative cost / break-even
	// above 100% because the other runner's simultaneous advance was
	// silently dropped.
	re24 := realRE24()
	be, reward, cost := BreakEvenRate(re24, "12_", 1, "2")
	if be < 0 || be > 1 {
		t.Fatalf("break-even = %v, want in [0, 1]", be)
	}
	if reward <= 0 {
		t.Fatalf("reward = %v, want > 0", reward)
	}
	if cost <= 0 {
		t.Fatalf("cost = %v, want > 0", cost)
	}
}
