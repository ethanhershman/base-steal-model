import { BASE_STATES, VALID_TARGETS } from "@/lib/types"
import type { BaseCode, Situation, Target } from "@/lib/types"

import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface SituationFormProps {
  value: Situation
  onChange: (patch: Partial<Situation>) => void
}

const BASE_CODE_LABELS: Record<BaseCode, string> = {
  "1__": "Runner on 1st",
  _2_: "Runner on 2nd",
  __3: "Runner on 3rd",
  "12_": "Runners on 1st & 2nd",
  "1_3": "Runners on 1st & 3rd",
  _23: "Runners on 2nd & 3rd",
  "123": "Bases loaded",
}

const TARGET_LABELS: Record<Target, string> = {
  "2": "2nd base",
  "3": "3rd base",
  H: "Home",
}

// The game-state half of the prediction request -- base_code/target/etc.
// Player stats are owned by App.tsx (filled in from PlayerCombobox
// selections), not this form.
export function SituationForm({ value, onChange }: SituationFormProps) {
  const validTargets = VALID_TARGETS[value.base_code]

  function handleBaseCodeChange(baseCode: BaseCode) {
    const targets = VALID_TARGETS[baseCode]
    onChange({
      base_code: baseCode,
      // Keep the current target if it's still valid for the new base
      // state (e.g. switching 1__ -> 12_ keeps "2"), otherwise fall back
      // to that state's first valid target.
      target: targets.includes(value.target) ? value.target : targets[0],
    })
  }

  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="inning">Inning</Label>
        <Input
          id="inning"
          type="number"
          min={1}
          max={20}
          value={value.inning}
          onChange={(e) => onChange({ inning: Number(e.target.value) })}
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <Label>Half</Label>
        <Select value={String(value.half)} onValueChange={(v: string) => onChange({ half: Number(v) as 0 | 1 })}>
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="0">Top</SelectItem>
            <SelectItem value="1">Bottom</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="flex flex-col gap-1.5">
        <Label>Outs</Label>
        <Select value={String(value.outs)} onValueChange={(v: string) => onChange({ outs: Number(v) })}>
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="0">0</SelectItem>
            <SelectItem value="1">1</SelectItem>
            <SelectItem value="2">2</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="score_diff">Score (batting team minus other)</Label>
        <Input
          id="score_diff"
          type="number"
          value={value.score_diff}
          onChange={(e) => onChange({ score_diff: Number(e.target.value) })}
        />
      </div>

      <div className="col-span-2 flex flex-col gap-1.5">
        <Label>Base state</Label>
        <Select value={value.base_code} onValueChange={(v: string) => handleBaseCodeChange(v as BaseCode)}>
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {BASE_STATES.map((bc) => (
              <SelectItem key={bc} value={bc}>
                {BASE_CODE_LABELS[bc]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex flex-col gap-1.5">
        <Label>Stealing</Label>
        <Select value={value.target} onValueChange={(v: string) => onChange({ target: v as Target })}>
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {validTargets.map((t) => (
              <SelectItem key={t} value={t}>
                {TARGET_LABELS[t]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex gap-4">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="balls">Balls</Label>
          <Input
            id="balls"
            type="number"
            min={0}
            max={3}
            value={value.balls}
            onChange={(e) => onChange({ balls: Number(e.target.value) })}
          />
        </div>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="strikes">Strikes</Label>
          <Input
            id="strikes"
            type="number"
            min={0}
            max={2}
            value={value.strikes}
            onChange={(e) => onChange({ strikes: Number(e.target.value) })}
          />
        </div>
      </div>

      <label className="col-span-2 flex items-center gap-2 text-sm">
        <Checkbox
          checked={value.is_double_steal}
          onCheckedChange={(checked: boolean | "indeterminate") =>
            onChange({ is_double_steal: checked === true })
          }
        />
        Double/triple steal (another runner breaking simultaneously)
      </label>
    </div>
  )
}
