import { useMutation } from "@tanstack/react-query"
import { useState } from "react"

import { PlayerCombobox } from "@/components/PlayerCombobox"
import { PlayerStatFields } from "@/components/PlayerStatFields"
import { ResultCard } from "@/components/ResultCard"
import { SituationForm } from "@/components/SituationForm"
import { predictStealDecision } from "@/lib/api"
import type {
  CatcherStats,
  PitcherStats,
  PlayerSearchResult,
  RunnerStats,
  Situation,
} from "@/lib/types"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { AnimatedGroup } from "@/components/ui/animated-group"

const DEFAULT_SITUATION: Situation = {
  inning: 9,
  half: 1,
  outs: 2,
  base_code: "1__",
  score_diff: -1,
  target: "2",
  balls: 0,
  strikes: 0,
  is_double_steal: false,
  runner_bats_lhb: false,
  pitcher_throws_lhp: false,
  runner_prior_sr: 0,
  runner_prior_att: 0,
  pitcher_prior_sr_allowed: 0,
  catcher_prior_cs_rate: 0,
  runner_sprint_speed: null,
  runner_age: null,
  catcher_pop_time: null,
}

export default function PredictorPage() {
  const [situation, setSituation] = useState<Situation>(DEFAULT_SITUATION)
  const [runner, setRunner] = useState<PlayerSearchResult | null>(null)
  const [pitcher, setPitcher] = useState<PlayerSearchResult | null>(null)
  const [catcher, setCatcher] = useState<PlayerSearchResult | null>(null)

  const mutation = useMutation({ mutationFn: predictStealDecision })

  function handleSituationChange(patch: Partial<Situation>) {
    setSituation((prev) => ({ ...prev, ...patch }))
  }

  function handleRunnerSelect(player: PlayerSearchResult) {
    const stats = player.stats as RunnerStats
    setRunner(player)
    setSituation((prev) => ({
      ...prev,
      runner_bats_lhb: stats.bats_lhb,
      runner_prior_sr: stats.prior_sr,
      runner_prior_att: stats.prior_att,
      runner_sprint_speed: stats.sprint_speed_missing ? null : stats.sprint_speed,
      runner_age: stats.age_missing ? null : stats.age,
    }))
  }

  function handlePitcherSelect(player: PlayerSearchResult) {
    const stats = player.stats as PitcherStats
    setPitcher(player)
    setSituation((prev) => ({
      ...prev,
      pitcher_throws_lhp: stats.throws_lhp,
      pitcher_prior_sr_allowed: stats.prior_sr_allowed,
    }))
  }

  function handleCatcherSelect(player: PlayerSearchResult) {
    const stats = player.stats as CatcherStats
    setCatcher(player)
    setSituation((prev) => ({
      ...prev,
      catcher_prior_cs_rate: stats.prior_cs_rate,
      catcher_pop_time: stats.pop_time_missing ? null : stats.pop_time,
    }))
  }

  return (
    <div className="mx-auto max-w-6xl p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold text-foreground">Should they steal?</h1>
        <p className="text-sm text-muted-foreground">
          Set the game situation and pick real players -- the model and decision layer trained on
          2023-2025 MLB play-by-play do the rest.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-2 lg:items-start">
        <AnimatedGroup preset="blur-slide" className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Situation</CardTitle>
            </CardHeader>
            <CardContent>
              <SituationForm value={situation} onChange={handleSituationChange} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Players</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div className="flex flex-col gap-1">
                <Label>Runner</Label>
                <PlayerCombobox role="runner" label="Runner" selected={runner} onSelect={handleRunnerSelect} />
                <PlayerStatFields role="runner" player={runner} />
              </div>
              <div className="flex flex-col gap-1">
                <Label>Pitcher</Label>
                <PlayerCombobox role="pitcher" label="Pitcher" selected={pitcher} onSelect={handlePitcherSelect} />
                <PlayerStatFields role="pitcher" player={pitcher} />
              </div>
              <div className="flex flex-col gap-1">
                <Label>Catcher</Label>
                <PlayerCombobox role="catcher" label="Catcher" selected={catcher} onSelect={handleCatcherSelect} />
                <PlayerStatFields role="catcher" player={catcher} />
              </div>
            </CardContent>
          </Card>

          <Button size="lg" onClick={() => mutation.mutate(situation)} disabled={mutation.isPending}>
            {mutation.isPending ? "Calculating..." : "Get recommendation"}
          </Button>

          {mutation.isError && (
            <p className="text-sm text-destructive">{(mutation.error as Error).message}</p>
          )}
        </AnimatedGroup>

        <div className="lg:sticky lg:top-6">
          {mutation.isSuccess ? (
            <ResultCard result={mutation.data} />
          ) : (
            <div className="flex min-h-48 flex-col items-center justify-center rounded-2xl border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
              Fill in the situation and players, then get a recommendation.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
