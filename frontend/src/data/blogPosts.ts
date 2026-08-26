export type BlogPost = {
  slug: string;
  title: string;
  excerpt: string;
  category: string;
  date: string;
  readingTime: string;
  image?: string;
  body: string[];
};

export const BLOG_POSTS: BlogPost[] = [
  {
    slug: "what-is-trendlens",
    title: "What is TrendLens, and why did we build it?",
    excerpt:
      "The origin story: a stubborn gap in every trend tool on Earth, two years of unnamed cottagecore, and the machine we built to close it.",
    category: "The project",
    date: "Jun 2, 2026",
    readingTime: "4 min read",
    body: [
      "Here's a party trick that fails every time. Ask Google Trends when an aesthetic started. It can't tell you — not because Google is slow, but because Google is deaf to anything without a word attached.",
      "And that's the strange truth about trends: they don't start with names. They start as *vibes*. Cottagecore was all over your feed for two years — sourdough starters, linen dresses, jam in tiny jars — before anyone agreed what to call it. For those two years it was invisible to every trend tool on Earth. If you need the word to find it, you're already late.",
      "That gap got stuck in our heads. What if you could spot a trend while it's still awkward to describe? So we built **TrendLens** — a machine that watches social media the way you do: with its eyes, not its ears.",
      "It doesn't read captions or count hashtags. It just notices when things look alike — same golden-hour glow, same messy-hands-holding-a-bowl energy, same suspiciously photogenic croissant. When enough pictures rhyme, they form a group. And when that group quietly snowballs from twelve photos to sixty while *nobody has said the word yet* — that's a trend in the making, and our dashboard lights up.",
      "Then we let you interrogate it. Type *what cafe aesthetic is rising right now?* into the Lens and it checks the actual posts before answering — you get real numbers, real growth curves and the real photos behind them. No evidence, no answer. No vibes-based claims about vibes.",
      "We won't pretend it's magic. When the machine labels a cluster 'minimalist latte art', that's its best guess from looking at pictures — like naming a cloud. The growth numbers are solid; the nicknames are a work in progress. We think the trade is worth it: it's the price of arriving before the word does.",
      "So that's TrendLens: a very patient pair of eyes that never scrolls past anything, watching for the next big look while it's still too weird to explain at a dinner party. Stick around — the best part of the feed hasn't been named yet.",
    ],
  },
];

export function getPost(slug: string): BlogPost | undefined {
  return BLOG_POSTS.find((p) => p.slug === slug);
}
