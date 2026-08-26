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
    slug: "before-the-hashtag",
    title: "Cottagecore existed for two years before it had a name",
    excerpt:
      "The clusters were visible in late 2017 — warm light, baking hands, linen dresses — but the word didn't land until 2019. What was the lens seeing in between?",
    category: "Field notes",
    date: "Aug 18, 2026",
    readingTime: "6 min read",
    image: "https://picsum.photos/seed/trendlens-blog-1/1200/700",
    body: [
      "Every trend tool on the market starts with language. Google Trends needs the query. Brandwatch needs the hashtag. Pinterest needs the search. If a look doesn't have words attached yet, it is invisible to all of them.",
      "But aesthetics don't start with names. They start as images — a cluster of posts that rhyme visually before anyone can say why. Cottagecore is the canonical example: the visual vocabulary existed for roughly two years before the word was coined and the hashtag took off.",
      "This is the gap TrendLens was built to watch. We embed every post through CLIP, cluster what comes out, and track which clusters are quietly growing. No keywords required.",
      "When we replayed historical data, the cottagecore cluster was detectable months before its naming moment — growing steadily under captions about baking, gardens and slow mornings that never mentioned it by name.",
      "The takeaway isn't that names don't matter. It's that by the time a name exists, the trend has already done most of its early spread. The interesting window is before the word.",
    ],
  },
  {
    slug: "anatomy-of-a-rising-cluster",
    title: "The anatomy of a rising visual cluster",
    excerpt:
      "Growth rate alone is a terrible trend signal. Here's how we combine growth, size and stability into an emerging score — and what each stage of the lifecycle looks like.",
    category: "How it works",
    date: "Aug 9, 2026",
    readingTime: "8 min read",
    image: "https://picsum.photos/seed/trendlens-blog-2/1200/700",
    body: [
      "A cluster is a group of images whose embeddings sit close together in CLIP space. Some clusters are permanent residents — coffee, sunsets, sneakers will never die. The interesting ones are the ones that move.",
      "We give every cluster three numbers: recent growth versus the prior window, total size, and stability across days. A spike of two posts means nothing. A steady climb over ten days from a small base is exactly what an emerging aesthetic looks like.",
      "Lifecycle labels follow from those numbers. Rising clusters grow fast from small bases. Stable clusters hold their share. Declining clusters fade — and watching what replaces them is half the fun.",
      "The honest caveat: cluster names come from captioning models, not humans. 'Minimalist latte art' is our best interpretation of a centroid, not ground truth. The metrics are real; the nouns are educated guesses.",
    ],
  },
  {
    slug: "why-engagement-lies",
    title: "Why engagement numbers lie about trends",
    excerpt:
      "A viral post gets likes because of the account, the algorithm, and the hour it posted — not just the aesthetic. Separating style signal from account noise.",
    category: "Opinion",
    date: "Jul 28, 2026",
    readingTime: "5 min read",
    body: [
      "Ask most social tools 'what's working?' and they'll sort posts by engagement. That answers a different question: which accounts are big right now.",
      "Engagement is heavily confounded by follower count, posting time and format. A mediocre photo from a two-million-follower account will outperform a brilliant one from a fresh page every single time.",
      "That's why TrendLens compares engagement within clusters, not across the whole feed. When every member of a cluster shares a visual grammar, differences in engagement start to say something about the grammar itself.",
      "It's not perfect — nothing is — but it's the difference between 'this account is popular' and 'this way of shooting is pulling ahead.'",
    ],
  },
  {
    slug: "reading-the-feed-like-a-forecaster",
    title: "Reading the feed like a forecaster",
    excerpt:
      "Four habits for spotting visual trends manually — useful even if you never open a dashboard. Spoiler: screenshot everything.",
    category: "Guides",
    date: "Jul 12, 2026",
    readingTime: "4 min read",
    image: "https://picsum.photos/seed/trendlens-blog-4/1200/700",
    body: [
      "You don't need embeddings to notice a trend forming. You need patience and a screenshots folder.",
      "Habit one: save images that feel familiar but that you can't attribute to a specific account. Familiarity without a source is often a trend mid-spread.",
      "Habit two: watch the edges of your feed, not the center. By the time your core timeline repeats a look, it has already peaked where it started.",
      "Habit three: track props and settings, not subjects. Latte art is eternal; the saucer it sits on changes yearly.",
      "Habit four: when you can finally name it, you're late. The whole point is noticing the pattern while it's still awkward to describe. If you have to say 'you know, that kind of…' — keep going. That hesitation is the signal.",
    ],
  },
];

export function getPost(slug: string): BlogPost | undefined {
  return BLOG_POSTS.find((p) => p.slug === slug);
}
