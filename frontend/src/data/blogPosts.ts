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
    slug: "desserts-on-the-feed",
    title: "Desserts on the feed: from micro-batch baking to viral cookie shops",
    excerpt:
      "How a cluster of prettiest-bake posts quietly grew while no one had agreed on a name — and what that told us about the way food trends travel.",
    category: "From the data",
    date: "Aug 20, 2026",
    readingTime: "5 min read",
    body: [
      "Every food aesthetic eventually gets a name. But long before the name, there's just a pile of pictures that suddenly all look the same — same crinkle-edged cookie, same swirl piped onto a sheet cake, same hands dusted in flour holding up a jar of jam.",
      "In this month's feed, desserts and micro-batch baking kept clustering together. Not because the machine reads recipes — it can't. Because the images rhyme: the same overhead angle, the same warm daylight, the same deliberate mess on the counter. When enough pictures rhyme, the cluster forms, and it forms *before* anyone is searching for the trend by name.",
      "The interesting part is where these bakes live. Cookie shops you'd never hear about through a search engine are all over the visual feed — folding boxes of soft-batch cookies, cream-cheese frosting smeared just-so, a perfect circle of crumbs. It's a vibe you can only catch with your eyes.",
      "We track this stuff because the trajectory is *usually* the same: a handful of posts, then a slow climb, then a tipping point where every account in the niche posts the same thing within a week. That tipping point is where interest converts to revenue — and it's exactly where word-based tools are still looking up the wrong keyword.",
      "None of this is magic. The machine's nickname for this cluster — 'dessert' — is the least interesting thing about it. What matters is the *shape* of the growth curve: micro-batch baking isn't loud. It's a slow, steady, photogenic hum. And the hum, unlike a fad, tends to stick around.",
    ],
  },
  {
    slug: "cottagecore-but-make-it-data",
    title: "Cottagecore, but make it data: how visual clusters become named aesthetics",
    excerpt:
      "Cottagecore lived in your feed for years before it had a name. Here's the anatomy of how an unnamed vibe becomes a searchable word — and how TrendLens watches the part that happens first.",
    category: "The project",
    date: "Aug 4, 2026",
    readingTime: "4 min read",
    body: [
      "Here's the party trick again: ask when an aesthetic *started* and no tool can really tell you. Not because the data isn't there, but because an aesthetic starts as a visual rhythm, not a word.",
      "Cottagecore is the cleanest example we have. For two years it existed as sourdough starters, linen dresses, jam in tiny jars, flowers pressed in old books — a whole visual language — long before anyone agreed on the label. The word arrived last. The vibe arrived first.",
      "But 'first' is doing a lot of work. What does first actually look like in the data? It looks like a small set of images that keep agreeing with each other. Same golden-hour light. Same worn wooden table. Same hands doing something slow and domestic. If you only counted hashtags or searched for the eventual name, you'd be standing exactly where every trend tool stands: late.",
      "TrendLens watches the part before the word. It looks at pictures and asks 'do these images rhyme?' When enough of them agree, they form a cluster. When the cluster grows quietly — twelve images becoming sixty while nobody has said the label yet — that's a named aesthetic in waiting.",
      "The naming comes later, and it's messy. 'Cottagecore' stuck because enough people pointed at the same look and settled. Our machine's best-guess nicknames are exactly that: a guess, like naming a cloud. Sometimes the nickname is great. Sometimes it's 'minimalist latte art' and we shrug. The growth numbers are the honest part; the name is a label we're still learning to write on.",
      "So when you see a trend bloom out of nowhere, remember it was probably never out of nowhere. It was a vibe, rhyming quietly, waiting for a word to catch up. And by the time the word catches up, we were already watching.",
    ],
  },
  {
    slug: "trouble-with-naming-clouds",
    title: "The trouble with naming clouds",
    excerpt:
      "When the machine calls a cluster 'minimalist latte art', that's a guess — a good guess, but a guess. Here's why we separate the honest growth numbers from the hopeful nicknames, and why it matters.",
    category: "The project",
    date: "Jul 9, 2026",
    readingTime: "4 min read",
    body: [
      "Open the Lens and ask what's rising. You'll get a cluster, a growth curve, and a nickname. The nickname is where people smile — and it's also the part we trust the least.",
      "When we say a cluster is 'minimalist latte art', we're not reading the room. We're looking at a group of images and our model is offering its best guess at what they have in common, the same way you'd point at a cloud and say 'that one looks like a whale.' Sometimes it's spot on. Sometimes it's embarrassingly off. It is never *proof*.",
      "So we built a separation of powers. The growth numbers — how many posts, how fast, how the curve bends — those we treat as solid. Someone, somewhere, actually posted those images; the counts are verifiable. The nickname, by contrast, is a hypothesis someone could politely disagree with.",
      "Why go to all this trouble instead of just slapping a confident label on everything? Because a confident label that's wrong is worse than a cautious one. If you act on 'all-in on minimal latte art' and the cluster was really about a specific cafe's lighting, you've just bet marketing budget on a cloud.",
      "Our dream is that the label catches up to the data — that as a cluster matures, the machine's nickname converges on whatever the internet actually calls it. Until then, we'd rather tell you what we're sure of, and be honest about what we're only guessing at. Evidence first. Nicknames after.",
    ],
  },
  {
    slug: "what-5000-instagram-photos-taught-us",
    title: "What 5,000 Instagram photos taught us this month",
    excerpt:
      "Real numbers from the live feed: which niches are climbing, which are flat, and what the shape of this month's images says about where attention is heading.",
    category: "From the data",
    date: "Jun 24, 2026",
    readingTime: "5 min read",
    body: [
      "Every month we walk through what the feed actually contained — not what we hoped it contained. This month's haul: a few thousand real posts pulled from public Instagram accounts, and a wall of images worth looking at properly.",
      "The headline is less glamorous than the demo but truer: food still owns the feed. A clear majority of the images we collected cluster around food and drinks — the baristas, the bakers, the overhead shots of something on a plate. It's not a surprise; food is the most photographed thing on the internet. But the *split* is worth noting: it's not all one aesthetic.",
      "Coffee is loud and consistent — latte art, tamping, the ritual of the shop — a steady rhythm of pictures that looks almost the same every week. That's the signature of a mature niche: lots of volume, very little novelty. Desserts and micro-batch baking sit in the opposite corner: smaller volume, but climbing, and far more visually exploratory. The newness lives where the count is still low.",
      "Fashion and photography make up another big slice, led by a handful of very active accounts. What's striking is how *concentrated* the attention is — a small number of accounts produce a huge share of the images. That's true on the visual feed and it's true everywhere else: a few creators set the look, and the long tail reshapes it.",
      "Beauty and skincare are present but quieter, still finding their visual register this month. Volume alone doesn't tell you much — a flat count can mean a niche at rest, and a climbing one can be a fad. The shape of the curve is the signal.",
      "None of this is a prediction. It's a portrait — an honest snapshot of what people couldn't stop posting. And the pattern we keep seeing is the same one that started it all: the growth shows up in the pictures long before it has a name.",
    ],
  },
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
