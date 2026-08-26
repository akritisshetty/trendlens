import PageTransition from "../components/navigation/PageTransition";
import Hero from "../components/hero/Hero";
import TrendsSection from "../components/trends/TrendsSection";
import LiveThemes from "../components/trends/LiveThemes";
import ThoughtSection from "../components/thoughts/ThoughtSection";
import FeedbackSection from "../components/feedback/FeedbackSection";

export default function Home() {
  return (
    <PageTransition>
      <Hero />
      <TrendsSection />
      <LiveThemes />
      <ThoughtSection />
      <FeedbackSection />
    </PageTransition>
  );
}
