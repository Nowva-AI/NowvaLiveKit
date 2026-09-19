import { Brain, Crosshair, Gauge, Volume2 } from "lucide-react";
import { Reveal } from "@/components/ui/Reveal";
import { SectionHeading } from "@/components/ui/SectionHeading";

const LOOP = [
  {
    icon: Crosshair,
    index: "01",
    title: "Detect",
    copy: "Rep 3: your knees cave 14 degrees. The engine grades the severity and traces the cause — weak glutes, stiff ankles — not just the symptom.",
  },
  {
    icon: Volume2,
    index: "02",
    title: "Cue",
    copy: "Nova says the fix out loud. The moment is logged: the fault, its severity, the diagnosed cause, and the exact cue given.",
  },
  {
    icon: Gauge,
    index: "03",
    title: "Measure",
    copy: "Next rep: the cave drops to 8 degrees. The same cameras that caught the fault grade the fix. Every cue gets a verdict — did it work?",
  },
  {
    icon: Brain,
    index: "04",
    title: "Learn",
    copy: "Across every rack, patterns emerge: which cue fixes which fault for which body. Tomorrow's coach is sharper than the one you trained with today.",
  },
] as const;

export function Flywheel() {
  return (
    <section id="flywheel" className="border-t border-border bg-bg-2">
      <div className="mx-auto max-w-6xl px-5 py-24 md:px-8 md:py-32">
        <SectionHeading
          eyebrow="The Flywheel"
          title={
            <>
              That{" "}
              <span className="gradient-text">self-improves.</span>
            </>
          }
          lead="Most fitness tech guesses whether its advice worked. Nova measures it — rep by rep, cue by cue — and every rack learns from the answer."
        />

        <div className="mt-16 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {LOOP.map((step, index) => (
            <Reveal key={step.index} delay={index * 0.08} className="h-full">
              <article className="card-lift group relative h-full overflow-hidden rounded-2xl border border-border bg-surface p-6">
                <div
                  aria-hidden
                  className="pointer-events-none absolute -right-16 -top-16 size-44 rounded-full opacity-0 transition-opacity duration-500 group-hover:opacity-100"
                  style={{
                    background:
                      "radial-gradient(circle, var(--glow), transparent 70%)",
                  }}
                />
                <div className="flex items-start justify-between">
                  <step.icon className="size-6 text-accent-ink" strokeWidth={1.6} />
                  <span className="font-mono text-xs text-fg-3">{step.index}</span>
                </div>
                <h3 className="mt-5 font-display text-lg font-bold tracking-tight text-fg">
                  {step.title}
                </h3>
                <p className="mt-3 text-sm leading-relaxed text-fg-2">{step.copy}</p>
              </article>
            </Reveal>
          ))}
        </div>

        <Reveal delay={0.2} className="mt-12">
          <span className="inline-flex items-center gap-2.5 rounded-full border border-accent/40 bg-accent/10 px-4 py-2">
            <Volume2 className="size-3.5 text-accent-ink" aria-hidden />
            <span className="font-mono text-sm text-accent-ink">
              &ldquo;Two weeks ago your knees caved on eight reps out of ten.
              Today: two.&rdquo;
            </span>
          </span>
        </Reveal>

        <Reveal delay={0.25} className="mt-8">
          <p className="max-w-2xl text-lg leading-relaxed text-fg-2">
            Every rep becomes a labeled data point — your proportions, your
            joint angles, the cue you heard, what happened next.{" "}
            <strong className="font-semibold text-fg">
              Closed-loop coaching data like this doesn&apos;t exist anywhere
              — not in labs, not in apps.
            </strong>{" "}
            And only the math travels: anonymized angles and outcomes make the
            fleet smarter, while your footage never leaves the rack.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
