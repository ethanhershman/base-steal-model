import type { PredictResponse } from "@/lib/types"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tilt } from "@/components/ui/tilt"
import { BorderTrail } from "@/components/ui/border-trail"
import { SlidingNumber } from "@/components/ui/sliding-number"

// sources is variable-length: 2 elements on the RE24 path, 3 on the
// win-probability path -- see backend/internal/decision/predict.go.
export function ResultCard({ result }: { result: PredictResponse }) {
  const isGo = result.decision === "GO"

  return (
    <Tilt rotationFactor={4} springOptions={{ stiffness: 200, damping: 20 }}>
      <Card className="relative overflow-hidden">
        <BorderTrail className={isGo ? "bg-success" : "bg-destructive"} size={90} />
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Recommendation</CardTitle>
          <Badge className={isGo ? "bg-success text-success-foreground" : "bg-destructive text-destructive-foreground"}>
            {result.decision}
          </Badge>
        </CardHeader>
        <CardContent className="flex flex-col gap-3 text-sm">
          <div className="grid grid-cols-2 gap-3">
            <Stat label="Model's predicted success" value={result.p_success} />
            <Stat label="Break-even needed" value={result.break_even} />
            <Stat label="Win probability now" value={result.win_prob_current} />
            <Stat label="Win probability if safe" value={result.win_prob_if_success} />
          </div>
          <p className="text-xs text-muted-foreground">
            Layer: {result.layer} · sample size{" "}
            {Number.isFinite(result.min_n) ? result.min_n : "large (certain outcome)"}
            {result.low_confidence && (
              <span className="ml-1 font-medium text-amber-600">
                -- low confidence, thin historical sample for this exact situation
              </span>
            )}
          </p>
          <p className="text-xs text-muted-foreground">Sources: {result.sources.join(", ")}</p>
        </CardContent>
      </Card>
    </Tilt>
  )
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="flex items-baseline gap-0.5 text-lg font-semibold">
        <SlidingNumber value={Number((value * 100).toFixed(1))} />
        <span>%</span>
      </div>
    </div>
  )
}
