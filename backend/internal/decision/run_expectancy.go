package decision

// baseOccupancy is a mutable scratch representation of a base_code used
// only inside the cascade logic below -- base_code strings are
// reconstructed via encode() once mutation is done.
type baseOccupancy struct {
	b1, b2, b3 bool
}

func newBaseOccupancy(baseCode string) baseOccupancy {
	return baseOccupancy{
		b1: baseCode[0] != '_',
		b2: baseCode[1] != '_',
		b3: baseCode[2] != '_',
	}
}

func (o baseOccupancy) encode() string {
	s := []byte("___")
	if o.b1 {
		s[0] = '1'
	}
	if o.b2 {
		s[1] = '2'
	}
	if o.b3 {
		s[2] = '3'
	}
	return string(s)
}

func (o *baseOccupancy) get(base int) bool {
	switch base {
	case 1:
		return o.b1
	case 2:
		return o.b2
	case 3:
		return o.b3
	}
	return false
}

func (o *baseOccupancy) set(base int, v bool) {
	switch base {
	case 1:
		o.b1 = v
	case 2:
		o.b2 = v
	case 3:
		o.b3 = v
	}
}

// cascadeFree ports run_expectancy._cascade_free: frees up `base` for an
// incoming runner. If it's already occupied -- only possible on a
// double/triple steal, where more than one runner breaks on the same pitch
// -- push that runner forward one base first (cascading further if THAT
// base is also occupied). Returns true if a run scored because a runner
// got pushed off 3rd.
//
// Recursive by design: free the destination first (which may itself
// recurse), only then clear the origin. Don't flatten this into a loop --
// the recursion order is what makes a triple-steal cascade correct.
func cascadeFree(o *baseOccupancy, base int) bool {
	if base > 3 || !o.get(base) {
		return false
	}
	scoredFurther := cascadeFree(o, base+1)
	if base+1 <= 3 {
		o.set(base+1, true)
		o.set(base, false)
		return scoredFurther
	}
	o.set(base, false)
	return true // this runner was pushed off 3rd and scored
}

var targetFromTo = map[string][2]int{
	"2": {1, 2},
	"3": {2, 3},
	"H": {3, 4},
}

// StateAfterSuccess ports run_expectancy._state_after_success. Returns the
// new base_code and how many runs scored as a side effect: the batting
// team's own steal-of-home run, plus any runner a double/triple steal
// cascade pushed off 3rd (see cascadeFree). These two run sources are
// independent and can both fire in principle -- don't special-case away
// the addition.
func StateAfterSuccess(baseCode, target string) (string, int) {
	o := newBaseOccupancy(baseCode)
	ft := targetFromTo[target]
	frm, to := ft[0], ft[1]

	cascadeScored := false
	if to <= 3 {
		cascadeScored = cascadeFree(&o, to)
		o.set(to, true)
	}
	o.set(frm, false)

	runs := 0
	if target == "H" {
		runs++
	}
	if cascadeScored {
		runs++
	}
	return o.encode(), runs
}

var targetFrom = map[string]int{"2": 1, "3": 2, "H": 3}

// StateAfterCaught ports run_expectancy._state_after_caught -- no cascade
// needed, removing a runner never collides with another.
func StateAfterCaught(baseCode, target string) string {
	o := newBaseOccupancy(baseCode)
	o.set(targetFrom[target], false)
	return o.encode()
}

// BreakEvenRate ports run_expectancy.break_even_rate. Returns (breakEven,
// reward, cost) where reward/cost are run-expectancy changes. Callers on
// the normal request path only invoke this after confirming
// (baseCode, outs) exists in re24 (see predict.go's layer routing).
func BreakEvenRate(re24 RE24Table, baseCode string, outs int, target string) (breakEven, reward, cost float64) {
	cur := re24[RE24Key{baseCode, outs}]
	succState, runBonus := StateAfterSuccess(baseCode, target)

	var reSucc float64
	if outs <= 2 {
		reSucc = float64(runBonus) + re24[RE24Key{succState, outs}]
	} else {
		reSucc = float64(runBonus)
	}

	var reCaught float64
	if outs >= 2 {
		reCaught = 0.0 // a caught stealing for the 3rd out ends the inning (RE=0)
	} else {
		caughtState := StateAfterCaught(baseCode, target)
		reCaught = re24[RE24Key{caughtState, outs + 1}]
	}

	reward = reSucc - cur
	cost = cur - reCaught
	denom := reward + cost
	if denom > 0 {
		breakEven = cost / denom
	} else {
		breakEven = 1.0
	}
	return breakEven, reward, cost
}
