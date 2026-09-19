import { Activity, CalendarRange, TrendingUp, UtensilsCrossed } from "lucide-react";
import { LogoWatermark } from "@/components/ui/LogoWatermark";
import { Reveal } from "@/components/ui/Reveal";
import { SectionHeading } from "@/components/ui/SectionHeading";

const PILLARS = [
  {
    icon: Activity,
    index: "01",
    title: "Form, corrected live",
    copy: "Nova watches every rep in 3D and speaks the fix before you finish it. Depth, knee track, bar path, trunk angle — measured, not guessed.",
  },
  {
    icon: CalendarRange,
    index: "02",
    title: "Training that adapts",
    copy: "Sets, loads, and progressions built from what actually happens under the bar. Stall on a lift and the plan changes. Recover fast and it pushes.",
  },
  {
    icon: UtensilsCrossed,
    index: "03",
    title: "Nutrition, dialed in",
    copy: "Calorie and protein targets matched to your training and your goal — cut, recomp, or grow — and adjusted as your numbers move.",
  },
  {
    icon: TrendingUp,
    index: "04",
    title: "Progress, on the record",
    copy: "Every rep, load, and PR logged automatically. No phone, no notebook. Ask Nova where you stand and get a straight answer.",
  },
] as const;

export function Coach() {
  return (
    <section id="coach" className="aurora-bg relative overflow-hidden">
      <LogoWatermark
        size={1000}
        className="left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 opacity-[0.025] dark:opacity-[0.03]"
      />
      <div className="relative mx-auto max-w-6xl px-5 py-24 md:px-8 md:py-36">
        <SectionHeading
          eyebrow="The Coach"
          title={
            <>
              Not a form checker.{" "}
              <span className="gradient-text">A full coach.</span>
            </>
          }
          lead="A great trainer notices things. The knee that flares up at depth, the rep you always rush when you're tired, the cue that finally makes it click. That's the attention and proximity that Nova is built to have."
        />

        <div className="mt-16 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {PILLARS.map((pillar, index) => (
            <Reveal key={pillar.index} delay={index * 0.08} className="h-full">
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
                  <pillar.icon className="size-6 text-accent-ink" strokeWidth={1.6} />
                  <span className="font-mono text-xs text-fg-3">{pillar.index}</span>
                </div>
                <h3 className="mt-5 font-display text-lg font-bold tracking-tight text-fg">
                  {pillar.title}
                </h3>
                <p className="mt-3 text-sm leading-relaxed text-fg-2">{pillar.copy}</p>
              </article>
            </Reveal>
          ))}
        </div>

        <Reveal delay={0.2} className="mt-10">
          <p className="max-w-2xl text-sm leading-relaxed text-fg-2">
            The form engine runs in our lab today — proven on the squat first.
            Programming and nutrition come with the membership; new movements
            ship as software updates.{" "}
            <strong className="font-semibold text-fg">
              One rack, the whole coach, improving for as long as you own it.
            </strong>
          </p>
        </Reveal>
      </div>
    </section>
  );
}
