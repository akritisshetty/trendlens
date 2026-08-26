import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";
import PageTransition from "../components/navigation/PageTransition";
import { BLOG_POSTS } from "../data/blogPosts";

export default function Blog() {
  const [featured, ...rest] = BLOG_POSTS;

  return (
    <PageTransition>
      <div className="mx-auto max-w-5xl px-5 pb-28 pt-28 md:px-8 md:pt-40">
        <header className="mb-14 md:mb-20">
          <p className="mb-3 text-xs uppercase tracking-[0.3em] text-ink-soft">
            The TrendLens journal
          </p>
          <h1 className="cropped-heading font-display text-[clamp(3rem,9vw,7rem)] font-bold">
            Notes from
            <br />the feed
          </h1>
        </header>

        {/* Featured post */}
        <motion.article
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.6 }}
          className="group border-y border-line py-10"
        >
          <Link to={`/blog/${featured.slug}`} className="block">
            <div className="grid gap-8 md:grid-cols-[2fr_1fr] md:items-end">
              <div>
                <p className="mb-4 text-xs uppercase tracking-[0.25em] text-accent">
                  Latest — {featured.category}
                </p>
                <h2 className="font-display text-3xl font-bold leading-tight transition-colors group-hover:text-accent md:text-5xl">
                  {featured.title}
                </h2>
                <p className="mt-4 max-w-xl text-ink-soft">{featured.excerpt}</p>
              </div>
              <div className="flex items-center justify-between text-sm text-ink-soft md:flex-col md:items-end md:gap-2">
                <span>{featured.date}</span>
                <span>{featured.readingTime}</span>
                <ArrowUpRight
                  className="h-5 w-5 transition-transform group-hover:-translate-y-1 group-hover:translate-x-1"
                  aria-hidden
                />
              </div>
            </div>
          </Link>
        </motion.article>

        {/* Rest of posts */}
        <div className="divide-y divide-line">
          {rest.map((post, i) => (
            <motion.article
              key={post.slug}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.5, delay: i * 0.06 }}
              className="group"
            >
              <Link
                to={`/blog/${post.slug}`}
                className="flex flex-wrap items-baseline justify-between gap-x-8 gap-y-2 py-8"
              >
                <div className="min-w-0 flex-1">
                  <span className="text-xs uppercase tracking-[0.25em] text-ink-soft">
                    {post.category}
                  </span>
                  <h3 className="mt-2 font-display text-2xl font-semibold leading-snug transition-colors group-hover:text-accent md:text-3xl">
                    {post.title}
                  </h3>
                  <p className="mt-2 line-clamp-2 max-w-2xl text-sm text-ink-soft">
                    {post.excerpt}
                  </p>
                </div>
                <span className="shrink-0 text-xs tabular-nums text-ink-soft">
                  {post.date} · {post.readingTime}
                </span>
              </Link>
            </motion.article>
          ))}
        </div>
      </div>
    </PageTransition>
  );
}
