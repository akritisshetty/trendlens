import PageTransition from "../components/navigation/PageTransition";
import Hero from "../components/hero/Hero";
import TrendsSection from "../components/trends/TrendsSection";
import InterestSelector from "../components/interests/InterestSelector";
import ThoughtSection from "../components/thoughts/ThoughtSection";
import FeedbackSection from "../components/feedback/FeedbackSection";

export default function Home() {
  return (
    <PageTransition>
      <Hero />
      <TrendsSection />
      <InterestSelector />
      <ThoughtSection />
      <FeedbackSection />
    </PageTransition>
  );
}
