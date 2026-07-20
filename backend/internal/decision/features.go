package decision

// BuildFeatureRow ports predict.py's _build_feature_row exactly, deriving
// the model's feature map from raw situation/player inputs the same way
// features.py derives it from real play-by-play.
func BuildFeatureRow(medians Medians, s Situation) map[string]float64 {
	// runner_on_third is NEVER a raw client field -- always derived from
	// base_code, then zeroed when stealing home (see predict.py: the
	// runner "on third" IS the runner attempting the steal in that case).
	runnerOnThird := s.BaseCode[2] != '_' && s.Target != "H"

	sprintSpeed, sprintMissing := resolveOrMedian(s.RunnerSprintSpeed, medians.RunnerSprintSpeed)
	age, ageMissing := resolveOrMedian(s.RunnerAge, medians.RunnerAge)
	popTime, popMissing := resolveOrMedian(s.CatcherPopTime, medians.CatcherPopTime)

	return map[string]float64{
		"steal_of_third":     boolToFloat(s.Target == "3"),
		"steal_of_home":      boolToFloat(s.Target == "H"),
		"is_double_steal":    boolToFloat(s.IsDoubleSteal),
		"runner_on_third":    boolToFloat(runnerOnThird),
		"late_inning":        boolToFloat(s.Inning >= 7),
		"outs":               float64(s.Outs),
		"balls":              float64(s.Balls),
		"strikes":            float64(s.Strikes),
		"score_diff":         float64(s.ScoreDiff),
		"close_game":         boolToFloat(absInt(s.ScoreDiff) <= 1),
		"runner_bats_lhb":    boolToFloat(s.RunnerBatsLHB),
		"pitcher_throws_lhp": boolToFloat(s.PitcherThrowsLHP),

		"runner_prior_sr":          s.RunnerPriorSR,
		"runner_prior_att":         float64(s.RunnerPriorAtt),
		"pitcher_prior_sr_allowed": s.PitcherPriorSRAllowed,
		"catcher_prior_cs_rate":    s.CatcherPriorCSRate,

		"runner_sprint_speed":         sprintSpeed,
		"runner_sprint_speed_missing": boolToFloat(sprintMissing),
		"runner_age":                  age,
		"runner_age_missing":          boolToFloat(ageMissing),
		"catcher_pop_time":            popTime,
		"catcher_pop_time_missing":    boolToFloat(popMissing),
	}
}

// resolveOrMedian: a nil pointer (client omitted/sent null) falls back to
// the training-set median with its _missing flag set -- exactly matching
// how features.py handles an unmatched Statcast join.
func resolveOrMedian(v *float64, median float64) (value float64, wasMissing bool) {
	if v == nil {
		return median, true
	}
	return *v, false
}

func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}
