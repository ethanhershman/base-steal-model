import { AnimatedGroup } from "@/components/ui/animated-group"
import { AnimatedNumber } from "@/components/ui/animated-number"
import { SlidingNumber } from "@/components/ui/sliding-number"
import { InView } from "@/components/ui/in-view"
import {
  BACKTEST_LAYERS,
  BACKTEST_TOTAL_ATTEMPTS,
  BACKTEST_HOLD_STORY,
} from "@/lib/backtest-data"

function PercentStat({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="flex items-baseline gap-0.5 text-2xl font-semibold text-foreground">
        <SlidingNumber value={Number((value * 100).toFixed(1))} />
        <span>%</span>
      </div>
    </div>
  )
}

function PolicyStat({
  label,
  value,
  accent,
}: {
  label: string
  value: number
  accent: "navy" | "red"
}) {
  return (
    <div
      className={`rounded-xl border p-4 ${
        accent === "navy"
          ? "border-brand-navy/20 bg-brand-navy/5"
          : "border-brand-red/20 bg-brand-red/5"
      }`}
    >
      <div className="text-xs text-muted-foreground">{label}</div>
      <div
        className={`text-2xl font-semibold ${
          accent === "navy" ? "text-brand-navy" : "text-brand-red"
        }`}
      >
        +<SlidingNumber value={value} decimalSeparator="." />
      </div>
      <div className="text-xs text-muted-foreground">runs / attempt</div>
    </div>
  )
}

export default function ModelPerformancePage() {
  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-16 px-6 py-16">
      <header className="text-center">
        <h1 className="text-3xl font-semibold tracking-tight text-foreground">
          Model Performance
        </h1>
        <p className="mt-3 text-muted-foreground">
          Every held-out attempt got a GO/HOLD call, then we compared the
          model's policy against what actually happened.
        </p>
        <InView once>
          <div className="mt-6 inline-flex items-baseline gap-2">
            <span className="text-4xl font-semibold text-foreground">
              <AnimatedNumber value={BACKTEST_TOTAL_ATTEMPTS} />
            </span>
            <span className="text-sm text-muted-foreground">
              held-out attempts, 2025-06-01 to 2025-11-01
            </span>
          </div>
        </InView>
      </header>

      <AnimatedGroup preset="blur-slide" className="flex flex-col gap-10">
        {BACKTEST_LAYERS.map((layer) => (
          <section
            key={layer.key}
            className="rounded-2xl border border-border bg-card p-8"
          >
            <div className="mb-6 flex flex-col gap-1">
              <h2 className="text-lg font-semibold text-foreground">
                {layer.label}
              </h2>
              <p className="text-sm text-muted-foreground">
                {layer.description}
              </p>
            </div>

            <div className="mb-6 grid grid-cols-3 gap-4">
              <div>
                <div className="text-xs text-muted-foreground">Attempts</div>
                <div className="text-2xl font-semibold text-foreground">
                  <AnimatedNumber value={layer.attempts} />
                </div>
              </div>
              <PercentStat label="GO success rate" value={layer.goSuccessRate} />
              <PercentStat
                label="HOLD success rate"
                value={layer.holdSuccessRate}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <PolicyStat
                label="Actual historical policy"
                value={layer.actualPolicy}
                accent="red"
              />
              <PolicyStat
                label="Model policy"
                value={layer.modelPolicy}
                accent="navy"
              />
            </div>
          </section>
        ))}
      </AnimatedGroup>

      <section className="rounded-2xl border border-border bg-secondary p-8">
        <h2 className="mb-3 text-lg font-semibold text-foreground">
          Where the value came from
        </h2>
        <p className="text-sm leading-relaxed text-muted-foreground">
          The model's calls added real value over the real historical policy
          in both regimes, mainly by avoiding the lowest-probability
          attempts. Of the{" "}
          <span className="font-medium text-foreground">
            {BACKTEST_HOLD_STORY.totalHeld}
          </span>{" "}
          RE24 situations it would have held,{" "}
          <span className="font-medium text-foreground">
            {BACKTEST_HOLD_STORY.caughtIfAttempted}
          </span>{" "}
          were actually caught stealing versus{" "}
          <span className="font-medium text-foreground">
            {BACKTEST_HOLD_STORY.missedOpportunities}
          </span>{" "}
          missed opportunities.
        </p>
      </section>
    </div>
  )
}
