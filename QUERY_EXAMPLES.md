# TrendLens — Query Examples

> **What can I ask TrendLens?**
>
> TrendLens detects **visual trends** from Instagram image clusters. It answers
> questions about what visual aesthetics are rising, what styles get engagement,
> and how to photograph subjects for maximum impact — all grounded in real
> Instagram data from ~80 monitored accounts across food, fashion, photography
> and beauty.

---

## 1. Trend Discovery

> "What is trending right now?"

```bash
python -m src.rag "What cafe items are trending?"
python -m src.rag "What food photography styles are trending on Instagram?"
python -m src.rag "What visual aesthetics are rising this week?"
python -m src.rag "What's hot in food content right now?"
python -m src.rag "What content patterns are emerging in cafe culture?"
python -m src.rag "What kind of latte art gets the most engagement?"
python -m src.rag "What food styles are going viral on Instagram?"
```

### By subject
```bash
python -m src.rag "What pastry trends are rising?"
python -m src.rag "What dessert styles are popular right now?"
python -m src.rag "What breakfast aesthetics are trending?"
python -m src.rag "What coffee presentation styles are hot?"
python -m src.rag "What healthy food visuals are trending?"
python -m src.rag "What vegan food aesthetics are popular?"
```

---

## 2. Visual Style Analysis

> "How should I style my photos?"

```bash
python -m src.rag "What visual styles get the most engagement?"
python -m src.rag "What lighting style works best for food photos?"
python -m src.rag "What colour palettes are trending in food photography?"
python -m src.rag "What composition styles are popular for cafe content?"
python -m src.rag "What background styles work for food flat lays?"
python -m src.rag "What plating styles get the highest engagement?"
```

---

## 3. Photography Advice

> "How do I shoot X?"

These queries trigger **advice mode** — TrendLens gives step-by-step photo
guides instead of cluster listings.

```bash
python -m src.rag "How should I photograph a latte for Instagram?"
python -m src.rag "How do I take a picture of a pastry that gets likes?"
python -m src.rag "What should a cake photo look like to maximize engagement?"
python -m src.rag "How to shoot a brunch spread for social media?"
python -m src.rag "What background should I use for food product photos?"
python -m src.rag "How should I frame a coffee cup for maximum reach?"
python -m src.rag "What angle works best for photographing desserts?"
python -m src.rag "How to style a flat lay of baked goods?"
python -m src.rag "What lighting setup gets the most engagement for food photos?"
```

---

## 4. Engagement & Performance

> "What gets the most likes?"

```bash
python -m src.rag "What food photos get the most engagement?"
python -m src.rag "Which visual styles perform best on Instagram?"
python -m src.rag "What content types get more views — reels or photos?"
python -m src.rag "What posting style gets the highest engagement?"
python -m src.rag "What hashtags are performing best for food content?"
```

---

## 5. Image Requests

> "Show me examples"

Add these keywords to any query to request representative images:

```bash
python -m src.rag "Show me examples of trending food photography"
python -m src.rag "What does rising cafe aesthetic look like? Show me pictures"
python -m src.rag "Give me representative images of popular dessert styles"
python -m src.rag "Display visual examples of latte art trends"
python -m src.rag "See photos of emerging food plating styles"
```

---

## 6. Comparative & Specific

> "How does X compare to Y?"

```bash
python -m src.rag "How does minimalist food styling compare to maximalist?"
python -m src.rag "What's different about trending cafe content vs restaurant content?"
python -m src.rag "Are video posts or photo posts trending more?"
python -m src.rag "What separates high-engagement food photos from low-engagement ones?"
```

---

## 7. Niche Exploration

> "What about [specific niche]?"

TrendLens monitors **food, fashion, photography and beauty** accounts on
Instagram. Example queries per niche:

| Niche | Example Query |
|-------|---------------|
| Coffee | "What latte art styles are trending?" |
| Pastries | "What croissant or donut styles are popular?" |
| Healthy food | "What smoothie bowl aesthetics are rising?" |
| Street food | "What street food photography styles are hot?" |
| Fashion / street style | "What street style aesthetics are trending on Instagram?" |
| Outfit styling | "What layering styles are getting engagement?" |
| Menswear | "What menswear styling trends are emerging?" |
| Photography (general) | "What editing and colour grading styles are trending in photography?" |
| Portrait / night photography | "What portrait lighting styles get the most engagement?" |
| Landscape / wildlife | "What wildlife and nature photography visuals are trending?" |
| Makeup | "What makeup looks are trending on Instagram?" |
| Skincare | "What skincare aesthetics are rising?" |
| Cross-niche | "What visual trends span both fashion and beauty content right now?" |

---

## Query Intent Routing

TrendLens automatically detects your intent and routes accordingly:

| Intent | Trigger Words | Response Style |
|--------|---------------|----------------|
| **Trend discovery** | trending, rising, hot, popular, viral, emerging | Clustered list of visual themes with growth data |
| **Advice** | how to shoot, how to photograph, what should it look like, maximize engagement | Step-by-step photo guide |
| **Image request** | show me, display, pictures, images, representative | Answer + supporting images |
| **Scope rejection** | (out-of-scope patterns — see below) | Polite refusal with suggestion |

---

## Out of Scope

TrendLens will **not** answer questions about:

- Programming, code, software development
- Math, equations, science facts
- Recipes, cooking instructions (it's about *visual trends*, not how to cook)
- News, sports, weather, stocks
- Translation, trivia, geography
- Health, legal, or financial advice
- Homework, academic questions

> **Tip:** If you get a refusal, try rephrasing your question as a
> *visual trend* question. Instead of "recipe for chocolate cake", ask
> "What chocolate cake visuals are trending on Instagram?"

---

## Data Sources

| Source | What it provides |
|--------|-----------------|
| **Instagram (Apify)** | Real posts from ~80 accounts across food, fashion, photography and beauty — images, captions, timestamps, likes, comments, views, hashtags |
| **Reddit** (optional) | Posts from `r/foodporn`, `r/coffee` |
| **Wikimedia Commons** (optional) | Images for `latte art`, `coffee`, `street food`, `breakfast` |

---

## Tips for Better Queries

1. **Be specific about the visual subject** — "What latte art styles are trending?" works better than "What's trending?"
2. **Use trend-intent words** — "trending", "rising", "hot", "popular", "emerging" trigger the Instagram data path
3. **Ask for photography advice** — "How should I photograph X?" triggers step-by-step guides
4. **Request images** — Add "show me" or "with pictures" to get representative images
5. **Specify the niche** — "What cafe aesthetics..." or "What dessert styles..." narrows the focus

---

_Last updated: 2026-08-24_
