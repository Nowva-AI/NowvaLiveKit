import { AudioLines, Gem, ScanEye, ShieldCheck, Sparkles } from "lucide-react";
import { Reveal } from "@/components/ui/Reveal";
import { SectionHeading } from "@/components/ui/SectionHeading";
import { RackExplorer } from "@/components/rack/RackExplorer";

const FEATURES = [
  {
    icon: ScanEye,
    index: "01",
    title: "It measures. It doesn't guess.",
    copy: "A camera array reconstructs your body and the bar in 3D — 21 tracked points, 30 times a second, every joint angle solved mid-rep. The same class of motion capture a biomechanics lab runs, machined into a rack.",
  },
  {
    icon: AudioLines,
    index: "02",
    title: "It coaches the rep, the set, the year",
    copy: "A fault caught mid-rep, the fix spoken before your next one — and that's just the fastest thing it does. Nova recaps every set and workout, tells you what to change and why, and tracks your progress across months. A coach with a memory, not a rep alarm.",
  },
  {
    icon: Sparkles,
    index: "03",
    title: "It calibrates to your body",
    copy: "Femur length, torso ratio, stance width — read in the first seconds of standing. Nova judges every rep against your anatomy's thresholds, not a template's.",
  },
  {
    icon: Gem,
    index: "04",
    title: "It gets smarter after you buy it",
    copy: "The rack you buy is the worst it will ever be. New movements and smarter coaching — learned from every rep the whole fleet lifts — ship as software updates. The steel stays; the coach inside it keeps improving.",
  },
] as const;

export function RackShowcase() {
  return (
    <section id="rack" className="relative">
      <div className="mx-auto max-w-6xl px-5 pt-24 md:px-8 md:pt-36">
        <SectionHeading
          eyebrow="The Rack"
          title={
            <>
              Everything your coach does.{" "}
              <span className="gradient-text">Built into the steel.</span>
            </>
          }
          lead="Premium steel. Cameras machined into the uprights. A voice that lives in the frame. Take it apart and see for yourself — every part is yours to inspect."
        />
      </div>

      <p className="mx-auto mt-10 max-w-6xl px-5 text-center font-mono text-[0.62rem] uppercase tracking-[0.22em] text-fg-3 md:px-8">
        Drag to orbit · tap any part
      </p>

      <div className="mt-6 md:mt-8">
        <RackExplorer />
      </div>

      <div className="mx-auto max-w-6xl px-5 pb-24 md:px-8 md:pb-36">
        <p className="text-center font-mono text-[0.65rem] tracking-[0.2em] text-fg-3">
          NV-01 · CAD ASSEMBLY — DESIGN EVOLVING
        </p>

        <div className="mt-20 grid gap-5 md:grid-cols-2">
          {FEATURES.map((feature, index) => (
            <Reveal key={feature.index} delay={index * 0.08}>
              <article className="card-lift group relative h-full overflow-hidden rounded-2xl border border-border bg-surface p-7">
                <div
                  aria-hidden
                  className="pointer-events-none absolute -right-16 -top-16 size-44 rounded-full opacity-0 transition-opacity duration-500 group-hover:opacity-100"
                  style={{
                    background:
                      "radial-gradient(circle, var(--glow), transparent 70%)",
                  }}
                />
                <div className="flex items-start justify-between">
                  <feature.icon className="size-6 text-accent-ink" strokeWidth={1.6} />
                  <span className="font-mono text-xs text-fg-3">{feature.index}</span>
                </div>
                <h3 className="mt-5 font-display text-xl font-bold tracking-tight text-fg">
                  {feature.title}
                </h3>
                <p className="mt-3 text-sm leading-relaxed text-fg-2">{feature.copy}</p>
              </article>
            </Reveal>
          ))}
        </div>

        <Reveal className="mt-14">
          <div className="gradient-border aurora-bg grid items-center gap-8 rounded-2xl p-8 md:grid-cols-[auto_1fr] md:p-10">
            <ShieldCheck
              className="size-12 text-accent-ink md:size-14"
              strokeWidth={1.4}
              aria-hidden
            />
            <div>
              <h3 className="font-display text-2xl font-bold tracking-tight text-fg">
                Fully Local Intelligence.
              </h3>
              <p className="mt-3 max-w-2xl leading-relaxed text-fg-2">
                The biomechanics engine runs in our lab today, processing
                every frame before the next one arrives. Every rack ships
                with a single embedded computer that runs the entire stack —
                multi-camera 3D reconstruction, inverse kinematics, real-time
                fault detection, and Nova&apos;s voice. No server, no cloud,
                no round-trip latency. Your footage never leaves the machine
                it was captured on.
              </p>
            </div>
          </div>
        </Reveal>

        <Reveal className="mt-12">
          <a
            href="#reserve"
            data-cta="rack-section"
            className="group inline-flex items-center gap-1.5 font-mono text-sm tracking-wide text-accent-ink transition-opacity hover:opacity-75"
          >
            Reserve the founding batch
            <span
              aria-hidden
              className="transition-transform duration-300 ease-out group-hover:translate-x-1"
            >
              →
            </span>
          </a>
        </Reveal>
      </div>
    </section>
  );
}
