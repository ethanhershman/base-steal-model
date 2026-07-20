// Package decision ports the Python decision layer (src/run_expectancy.py,
// src/win_probability.py, src/train.py, src/predict.py) to Go, so the web
// app's backend needs no Python process at request time. See
// /Users/colin/.claude/plans/go-with-chi-and-soft-grove.md for the full
// design and why each porting choice was made.
package decision

import (
	"encoding/json"
	"math"
)

// BaseStates mirrors run_expectancy.BASE_STATES.
var BaseStates = []string{"___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"}

const MinCellN = 20 // win_probability.MIN_CELL_N
const ScoreClip = 4 // win_probability.SCORE_CLIP
const MaxInning = 9 // win_probability.MAX_INNING

type RE24Key struct {
	BaseCode string
	Outs     int
}

type WinProbKey struct {
	InningBucket int
	Half         int
	Outs         int
	BaseCode     string
	ScoreBucket  int
}

type WinProbCell struct {
	WinRate float64
	N       int
}

type RE24Table map[RE24Key]float64
type WinProbTable map[WinProbKey]WinProbCell

// Tables bundles the three lookup structures a prediction needs -- loaded
// once at server startup (see tables.go), never rebuilt per-request.
type Tables struct {
	RE24    RE24Table
	WPTable WinProbTable // after-success/after-caught (post-rule-change seasons)
	WPHold  WinProbTable // hold-only baseline (13 seasons)
}

// Medians are the training-set fallback values for missing Statcast inputs.
type Medians struct {
	RunnerSprintSpeed float64
	RunnerAge         float64
	CatcherPopTime    float64
}

// Model is a plain logistic regression: p = sigmoid(intercept + coef . x),
// on raw unscaled feature values (see model.go).
type Model struct {
	FeatureOrder []string
	Coefficients []float64
	Intercept    float64
}

// Situation mirrors predict.py's predict_steal_decision kwargs, minus
// runner_on_third (always derived server-side from BaseCode -- see
// features.go -- never accepted as a raw client field).
type Situation struct {
	Inning        int
	Half          int // 0 = top, 1 = bottom
	Outs          int
	BaseCode      string
	ScoreDiff     int
	Target        string // "2", "3", "H"
	Balls         int
	Strikes       int
	IsDoubleSteal bool

	RunnerBatsLHB         bool
	PitcherThrowsLHP      bool
	RunnerPriorSR         float64
	RunnerPriorAtt        int
	PitcherPriorSRAllowed float64
	CatcherPriorCSRate    float64

	RunnerSprintSpeed *float64
	RunnerAge         *float64
	CatcherPopTime    *float64
}

// MinN wraps a sample-size count that can be +Inf (win_prob_lookup's
// "certain" case -- see win_probability.go). encoding/json errors on raw
// +Inf, so this renders it as a large finite sentinel instead of failing
// to marshal the response.
type MinN float64

func (m MinN) MarshalJSON() ([]byte, error) {
	if math.IsInf(float64(m), 1) {
		return []byte("999999999"), nil
	}
	return json.Marshal(float64(m))
}

// Decision mirrors predict.py's predict_steal_decision returned dict
// field-for-field.
type Decision struct {
	Inning    int    `json:"inning"`
	Half      int    `json:"half"`
	Outs      int    `json:"outs"`
	BaseCode  string `json:"base_code"`
	ScoreDiff int    `json:"score_diff"`
	Target    string `json:"target"`

	WinProbCurrent   float64  `json:"win_prob_current"`
	WinProbIfSuccess float64  `json:"win_prob_if_success"`
	BreakEven        float64  `json:"break_even"`
	PSuccess         float64  `json:"p_success"`
	DecisionLabel    string   `json:"decision"`
	Layer            string   `json:"layer"`
	MinN             MinN     `json:"min_n"`
	LowConfidence    bool     `json:"low_confidence"`
	Sources          []string `json:"sources"`
}
