import type {
  CatcherStats,
  PitcherStats,
  PlayerRole,
  PlayerSearchResult,
  RunnerStats,
} from "@/lib/types"

interface PlayerStatFieldsProps {
  role: PlayerRole
  player: PlayerSearchResult | null
}

// Read-only summary of the selected player's auto-filled stats -- no
// manual override inputs, per the product decision to search real players
// rather than hand-enter stats.
export function PlayerStatFields({ role, player }: PlayerStatFieldsProps) {
  if (!player) return null

  if (role === "runner") {
    const s = player.stats as RunnerStats
    return (
      <p className="text-xs text-muted-foreground">
        {s.bats_lhb ? "Bats L" : "Bats R"} · prior success {(s.prior_sr * 100).toFixed(0)}%
        {" "}({s.prior_att} attempts) ·{" "}
        {s.sprint_speed_missing ? "sprint speed unknown" : `${s.sprint_speed?.toFixed(1)} ft/s`} ·{" "}
        {s.age_missing ? "age unknown" : `age ${s.age?.toFixed(0)}`}
      </p>
    )
  }

  if (role === "pitcher") {
    const s = player.stats as PitcherStats
    return (
      <p className="text-xs text-muted-foreground">
        {s.throws_lhp ? "Throws L" : "Throws R"} · prior steal-success rate allowed{" "}
        {(s.prior_sr_allowed * 100).toFixed(0)}%
      </p>
    )
  }

  const s = player.stats as CatcherStats
  return (
    <p className="text-xs text-muted-foreground">
      prior caught-stealing rate {(s.prior_cs_rate * 100).toFixed(0)}% ·{" "}
      {s.pop_time_missing ? "pop time unknown" : `pop time ${s.pop_time?.toFixed(2)}s`}
    </p>
  )
}
