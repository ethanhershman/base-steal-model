// Mirrors backend/internal/api/dto.go and backend/internal/decision/types.go
// field-for-field.

export const BASE_STATES = [
  "1__",
  "_2_",
  "__3",
  "12_",
  "1_3",
  "_23",
  "123",
] as const
export type BaseCode = (typeof BASE_STATES)[number]

export type Target = "2" | "3" | "H"

// Which steal targets make physical sense from each base state -- mirrors
// src/export_golden_fixtures.py's VALID_TARGETS.
export const VALID_TARGETS: Record<BaseCode, Target[]> = {
  "1__": ["2"],
  _2_: ["3"],
  __3: ["H"],
  "12_": ["2", "3"],
  "1_3": ["2", "H"],
  _23: ["3", "H"],
  "123": ["2", "3", "H"],
}

export interface Situation {
  inning: number
  half: 0 | 1
  outs: number
  base_code: BaseCode
  score_diff: number
  target: Target
  balls: number
  strikes: number
  is_double_steal: boolean

  runner_bats_lhb: boolean
  pitcher_throws_lhp: boolean
  runner_prior_sr: number
  runner_prior_att: number
  pitcher_prior_sr_allowed: number
  catcher_prior_cs_rate: number

  runner_sprint_speed: number | null
  runner_age: number | null
  catcher_pop_time: number | null
}

export interface PredictResponse {
  inning: number
  half: number
  outs: number
  base_code: string
  score_diff: number
  target: string

  win_prob_current: number
  win_prob_if_success: number
  break_even: number
  p_success: number
  decision: "GO" | "HOLD"
  layer: string
  min_n: number
  low_confidence: boolean
  sources: string[]
}

export type PlayerRole = "runner" | "pitcher" | "catcher"

export interface RunnerStats {
  bats_lhb: boolean
  prior_sr: number
  prior_att: number
  sprint_speed: number | null
  sprint_speed_missing: boolean
  age: number | null
  age_missing: boolean
}

export interface PitcherStats {
  throws_lhp: boolean
  prior_sr_allowed: number
}

export interface CatcherStats {
  prior_cs_rate: number
  pop_time: number | null
  pop_time_missing: boolean
}

export type PlayerStats = RunnerStats | PitcherStats | CatcherStats

export interface PlayerSearchResult<S extends PlayerStats = PlayerStats> {
  id: string
  name: string
  team: string | null
  stats: S
}
