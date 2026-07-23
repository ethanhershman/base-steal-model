import { NavLink } from "react-router"
import { Button } from "@/components/ui/button"
import { TextEffect } from "@/components/ui/text-effect"
import { TextLoop } from "@/components/ui/text-loop"
import { AnimatedGroup } from "@/components/ui/animated-group"
import { GlowEffect } from "@/components/ui/glow-effect"
import { Spotlight } from "@/components/ui/spotlight"
import { BorderTrail } from "@/components/ui/border-trail"
import { InView } from "@/components/ui/in-view"
import { SlidingNumber } from "@/components/ui/sliding-number"
import { DiamondField } from "@/components/graphics/DiamondField"
import { BACKTEST_LAYERS } from "@/lib/backtest-data"

const EXAMPLE_SITUATIONS = [
  "Runner on 1st, 2 outs, down 1, top 9th",
  "Runner on 2nd, 0 outs, tied, bottom 7th",
  "Runner on 1st & 2nd, 1 out, up 3, 5th inning",
]

const HOW_IT_WORKS = [
  {
    step: "1",
    title: "Set the situation",
    body: "Inning, outs, count, base state, and who's on the mound and behind the plate.",
  },
  {
    step: "2",
    title: "Pick real players",
    body: "Runner speed, pitcher hold times, catcher pop time -- pulled from 2023-2025 Statcast data.",
  },
  {
    step: "3",
    title: "Get a GO or HOLD call",
    body: "The model's predicted success rate is checked against the situation's break-even rate.",
  },
]

const previewStat = BACKTEST_LAYERS[0]

export default function HomePage() {
  return (
    <div className="flex flex-col gap-24 px-6 py-16">
      <section className="mx-auto flex max-w-5xl flex-col items-center gap-8 text-center">
        <div className="relative w-full overflow-hidden rounded-3xl border border-border bg-gradient-to-br from-secondary to-background px-8 py-20">
          <BorderTrail className="bg-brand-red" size={120} />
          <Spotlight className="from-brand-navy/30 via-brand-navy/10 to-transparent" size={300} />
          <DiamondField className="pointer-events-none absolute inset-0 mx-auto h-full w-full max-w-md text-brand-navy/10" />
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center opacity-20">
            <div className="relative h-40 w-40">
              <GlowEffect
                mode="breathe"
                colors={["var(--brand-navy)", "var(--brand-red)"]}
                blur="strong"
              />
            </div>
          </div>

          <div className="relative z-10 flex flex-col items-center gap-6">
            <TextEffect
              as="h1"
              per="word"
              preset="fade-in-blur"
              className="text-4xl font-semibold tracking-tight text-foreground sm:text-6xl"
            >
              Should the runner go?
            </TextEffect>
            <p className="max-w-xl text-balance text-base text-muted-foreground sm:text-lg">
              A success-probability model and break-even decision layer, trained
              on real MLB play-by-play, that tells you whether a stolen-base
              attempt is worth it.
            </p>

            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <span>e.g.</span>
              <TextLoop className="font-medium text-foreground">
                {EXAMPLE_SITUATIONS.map((s) => (
                  <span key={s}>{s}</span>
                ))}
              </TextLoop>
            </div>

            <Button asChild size="lg">
              <NavLink to="/predictor">Get a recommendation</NavLink>
            </Button>
          </div>
        </div>
      </section>

      <section className="mx-auto w-full max-w-5xl">
        <h2 className="mb-8 text-center text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          How it works
        </h2>
        <AnimatedGroup
          preset="blur-slide"
          className="grid gap-6 sm:grid-cols-3"
        >
          {HOW_IT_WORKS.map((item) => (
            <div
              key={item.step}
              className="rounded-2xl border border-border bg-card p-6"
            >
              <div className="mb-3 flex h-8 w-8 items-center justify-center rounded-full bg-primary text-sm font-semibold text-primary-foreground">
                {item.step}
              </div>
              <h3 className="mb-1 font-semibold text-foreground">
                {item.title}
              </h3>
              <p className="text-sm text-muted-foreground">{item.body}</p>
            </div>
          ))}
        </AnimatedGroup>
      </section>

      <section className="mx-auto w-full max-w-3xl">
        <InView once viewOptions={{ margin: "-100px" }}>
          <div className="rounded-2xl border border-border bg-gradient-to-br from-brand-navy-deep to-brand-navy p-10 text-center text-primary-foreground">
            <p className="mb-2 text-sm uppercase tracking-wide text-primary-foreground/70">
              Backtested against 2,714 real outcomes
            </p>
            <div className="flex items-center justify-center gap-1 text-5xl font-semibold">
              <span>+</span>
              <SlidingNumber value={previewStat.modelPolicy} decimalSeparator="." />
            </div>
            <p className="mt-2 text-sm text-primary-foreground/70">
              expected runs added per attempt over the real historical policy,
              RE24 situations
            </p>
            <Button asChild variant="secondary" className="mt-6">
              <NavLink to="/model-performance">See the full backtest</NavLink>
            </Button>
          </div>
        </InView>
      </section>
    </div>
  )
}
