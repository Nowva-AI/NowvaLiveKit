import { IntroStage } from "@/components/intro/IntroStage";
import { IntroLoader } from "@/components/layout/IntroLoader";
import { Navbar } from "@/components/layout/Navbar";
import { ScrollProgress } from "@/components/layout/ScrollProgress";
import { Footer } from "@/components/layout/Footer";
import { Hero } from "@/components/sections/Hero";
import { Mission } from "@/components/sections/Mission";
import { RackShowcase } from "@/components/sections/RackShowcase";
import { Coach } from "@/components/sections/Coach";
import { Flywheel } from "@/components/sections/Flywheel";
import { CtaBand } from "@/components/sections/CtaBand";
import { Manifesto } from "@/components/sections/Manifesto";
import { Pricing } from "@/components/sections/Pricing";
import { Faq } from "@/components/sections/Faq";
import { DELIVERY } from "@/lib/constants";

export default function Home() {
  return (
    <>
      {/* Homepage-only 3D intro. It must stay OUTSIDE #top: on takeover
          IntroStage inerts every direct child of <body>, and Providers
          renders no DOM element, so fragment children here land directly
          under <body> — keeping the overlay it later mounts out of the
          inerted #top subtree (its skip button must stay interactive).
          IntroLoader (the CSS splash) is homepage-only for the same reason
          legal pages shouldn't flash a brand intro. */}
      <IntroLoader />
      <IntroStage />
      <div id="top">
        <a href="#main" className="skip-link">
          Skip to content
        </a>
        <ScrollProgress />
        <Navbar />
        <main id="main">
          <Hero />
          <Mission />
          <RackShowcase />
          <Coach />
          <Flywheel />
          <Manifesto />
          <Pricing />
          <Faq />
          <CtaBand
            headline={
              <>
                Train with <span className="gradient-text">intelligence.</span>
              </>
            }
            sub={`Founding batch · ${DELIVERY} · $0 to reserve`}
            location="final-band"
          />
        </main>
        <Footer />
      </div>
    </>
  );
}
