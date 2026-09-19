import { LogoWatermark } from "@/components/ui/LogoWatermark";
import { Reveal } from "@/components/ui/Reveal";
import { SectionHeading } from "@/components/ui/SectionHeading";

const PARAGRAPHS = [
  "Elite athletes get a coach who watches every rep, programs every week, and dials in every meal. Everyone else gets an app that can't see them. That gap — between the coaching a few can afford and the coaching everyone deserves — is the unfairness we set out to fix.",
  "The only honest way to fix it is the full stack: a biomechanics engine that actually understands movement, the cameras and steel it lives in, and edge AI so it's instant and private. No shortcuts, no cloud, no bolt-on webcam.",
  "This product was impossible three years ago. Then edge AI crossed a threshold: a chip the size of a paperback can now run lab-grade computer vision in real time, no server behind it. The moment that became true, great coaching stopped being a labor problem and became a software problem. We got there first.",
  "Our goal is the fittest generation humanity has ever produced. When world-class coaching is available to everyone — not just the few who can pay for it — that's exactly what happens.",
] as const;

export function Manifesto() {
  return (
    <section id="manifesto" className="aurora-bg relative overflow-hidden">
      <LogoWatermark className="-right-24 top-1/2 -translate-y-1/2" />

      <div className="mx-auto max-w-6xl px-5 py-24 md:px-8 md:py-36">
        <SectionHeading
          eyebrow="Manifesto"
          title={
            <>
              Coaching this good shouldn&apos;t cost{" "}
              <span className="gradient-text">$25,000 a year.</span>
            </>
          }
        />
        <div className="mt-10 max-w-2xl space-y-6">
          {PARAGRAPHS.map((paragraph, index) => (
            <Reveal key={index} delay={index * 0.1}>
              <p
                className={
                  index === 0
                    ? "font-display text-xl font-medium leading-snug text-fg md:text-2xl"
                    : "text-lg leading-relaxed text-fg-2"
                }
              >
                {paragraph}
              </p>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
