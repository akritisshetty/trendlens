import PageTransition from "../components/navigation/PageTransition";
import Hero from "../components/hero/Hero";
import TrendsSection from "../components/trends/TrendsSection";
import HowItWorks from "../components/home/HowItWorks";
import ExploreSection from "../components/home/ExploreSection";
import CtaSection from "../components/home/CtaSection";
import FeedbackSection from "../components/feedback/FeedbackSection";

export default function Home() {
  return (
    <PageTransition>
      <Hero />
      <TrendsSection />
      <HowItWorks />
      <ExploreSection />
      <CtaSection />
      <FeedbackSection />
    </PageTransition>
  );
}
