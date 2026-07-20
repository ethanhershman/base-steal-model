package api

import (
	"strings"

	"basestealmodel/backend/internal/decision"
)

// predictRequest mirrors predict.py's predict_steal_decision kwargs, minus
// runner_on_third (always derived server-side from base_code -- see
// decision.BuildFeatureRow -- never accepted as a raw client field).
type predictRequest struct {
	Inning        int    `json:"inning"`
	Half          int    `json:"half"`
	Outs          int    `json:"outs"`
	BaseCode      string `json:"base_code"`
	ScoreDiff     int    `json:"score_diff"`
	Target        string `json:"target"`
	Balls         int    `json:"balls"`
	Strikes       int    `json:"strikes"`
	IsDoubleSteal bool   `json:"is_double_steal"`

	RunnerBatsLHB         bool    `json:"runner_bats_lhb"`
	PitcherThrowsLHP      bool    `json:"pitcher_throws_lhp"`
	RunnerPriorSR         float64 `json:"runner_prior_sr"`
	RunnerPriorAtt        int     `json:"runner_prior_att"`
	PitcherPriorSRAllowed float64 `json:"pitcher_prior_sr_allowed"`
	CatcherPriorCSRate    float64 `json:"catcher_prior_cs_rate"`

	RunnerSprintSpeed *float64 `json:"runner_sprint_speed"`
	RunnerAge         *float64 `json:"runner_age"`
	CatcherPopTime    *float64 `json:"catcher_pop_time"`
}

func (r predictRequest) toSituation() decision.Situation {
	return decision.Situation{
		Inning: r.Inning, Half: r.Half, Outs: r.Outs, BaseCode: r.BaseCode,
		ScoreDiff: r.ScoreDiff, Target: r.Target, Balls: r.Balls, Strikes: r.Strikes,
		IsDoubleSteal:         r.IsDoubleSteal,
		RunnerBatsLHB:         r.RunnerBatsLHB,
		PitcherThrowsLHP:      r.PitcherThrowsLHP,
		RunnerPriorSR:         r.RunnerPriorSR,
		RunnerPriorAtt:        r.RunnerPriorAtt,
		PitcherPriorSRAllowed: r.PitcherPriorSRAllowed,
		CatcherPriorCSRate:    r.CatcherPriorCSRate,
		RunnerSprintSpeed:     r.RunnerSprintSpeed,
		RunnerAge:             r.RunnerAge,
		CatcherPopTime:        r.CatcherPopTime,
	}
}

var validTargets = map[string]bool{"2": true, "3": true, "H": true}

// validate returns a non-empty error message if the request is malformed.
func (r predictRequest) validate() string {
	if r.Half != 0 && r.Half != 1 {
		return "half must be 0 or 1"
	}
	if r.Outs < 0 || r.Outs > 2 {
		return "outs must be 0, 1, or 2"
	}
	if !validTargets[r.Target] {
		return `target must be "2", "3", or "H"`
	}
	if !isValidBaseCode(r.BaseCode) {
		return "base_code must be one of " + strings.Join(decision.BaseStates, ", ")
	}
	return ""
}

func isValidBaseCode(bc string) bool {
	for _, s := range decision.BaseStates {
		if s == bc {
			return true
		}
	}
	return false
}

// playerSearchResult is the shape both /api/players/search's runner,
// pitcher, and catcher results share -- Stats holds the role-specific
// fields directly (embedded, no second round-trip needed for autofill).
type playerSearchResult struct {
	ID    string         `json:"id"`
	Name  string         `json:"name"`
	Team  *string        `json:"team"`
	Stats map[string]any `json:"stats"`
}
