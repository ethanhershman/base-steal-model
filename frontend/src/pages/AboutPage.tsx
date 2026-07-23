import { NavLink } from "react-router"
import { Button } from "@/components/ui/button"
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from "@/components/ui/accordion"

const FAQ_ITEMS = [
  {
    value: "data",
    question: "What data does it use?",
    answer:
      "Retrosheet play-by-play and Statcast skill tables from 2023-2025 -- runner sprint speed and age, pitcher hold/pickoff tendencies, catcher pop time and caught-stealing rate, plus the full game situation (inning, outs, count, base state, score).",
  },
  {
    value: "break-even",
    question: "What's a break-even rate?",
    answer:
      "The success rate a steal attempt needs to clear for it to be worth the risk, given what's on the line. Below break-even, the expected cost of getting caught outweighs the expected gain of being safe.",
  },
  {
    value: "why-hold",
    question: "Why isn't every high-probability steal a GO?",
    answer:
      "A high success rate isn't enough on its own -- it has to clear the break-even bar for that specific situation. Late and close games raise the bar sharply, since a caught stealing can end a trailing team's last real chance.",
  },
]

export default function AboutPage() {
  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-10 px-6 py-16">
      <header>
        <h1 className="text-3xl font-semibold tracking-tight text-foreground">
          About this project
        </h1>
        <p className="mt-4 leading-relaxed text-muted-foreground">
          This tool predicts whether a base runner should attempt a steal in
          a given situation. It's an independent research project and is not
          affiliated with, endorsed by, or sponsored by MLB or any team.
        </p>
      </header>

      <section>
        <h2 className="mb-3 text-xl font-semibold text-foreground">
          How the model works
        </h2>
        <div className="flex flex-col gap-4 leading-relaxed text-muted-foreground">
          <p>
            <span className="font-medium text-foreground">
              Success-probability model.
            </span>{" "}
            An XGBoost model trained on runner, pitcher, catcher, and
            situation features predicts the probability that a given steal
            attempt succeeds.
          </p>
          <p>
            <span className="font-medium text-foreground">
              Break-even decision layer.
            </span>{" "}
            That probability is compared against a situational break-even
            rate derived from run expectancy (RE24) for most of the game.
            Late and close situations switch to a win-probability framework
            instead, since RE24's run-based math badly understates the cost
            of a caught stealing that ends a trailing team's last chance.
          </p>
        </div>
      </section>

      <section className="rounded-2xl border border-border bg-secondary p-6">
        <h2 className="mb-2 text-lg font-semibold text-foreground">
          Validated against real outcomes
        </h2>
        <p className="mb-4 text-sm leading-relaxed text-muted-foreground">
          The model was backtested against 2,714 held-out real steal attempts
          from the 2025 season -- not a hand-picked sample.
        </p>
        <Button asChild variant="outline" size="sm">
          <NavLink to="/model-performance">See the full backtest</NavLink>
        </Button>
      </section>

      <section>
        <h2 className="mb-3 text-xl font-semibold text-foreground">
          FAQ
        </h2>
        <Accordion className="flex flex-col" transition={{ type: "spring", stiffness: 220, damping: 26 }}>
          {FAQ_ITEMS.map((item) => (
            <AccordionItem
              key={item.value}
              value={item.value}
              className="border-b border-border py-3"
            >
              <AccordionTrigger className="flex w-full items-center justify-between text-left text-sm font-medium text-foreground">
                {item.question}
              </AccordionTrigger>
              <AccordionContent className="pt-2 text-sm text-muted-foreground">
                {item.answer}
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </section>
    </div>
  )
}
