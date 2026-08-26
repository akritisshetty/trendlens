import { Link, useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft } from "lucide-react";
import PageTransition from "../components/navigation/PageTransition";
import { getPost } from "../data/blogPosts";

export default function BlogPost() {
  const { slug } = useParams();
  const post = slug ? getPost(slug) : undefined;

  if (!post) {
    return (
      <PageTransition>
        <div className="mx-auto flex min-h-screen max-w-2xl flex-col items-center justify-center gap-6 px-6 text-center">
          <h1 className="font-display text-4xl font-bold">Post not found</h1>
          <Link
            to="/blog"
            className="flex items-center gap-2 text-sm underline decoration-line underline-offset-4 hover:decoration-accent"
          >
            <ArrowLeft className="h-4 w-4" aria-hidden /> Back to all posts
          </Link>
        </div>
      </PageTransition>
    );
  }

  return (
    <PageTransition>
      <article className="mx-auto max-w-2xl px-6 pb-28 pt-28 md:pt-36">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55 }}
        >
          <Link
            to="/blog"
            className="mb-12 inline-flex items-center gap-2 text-sm text-ink-soft transition-colors hover:text-ink"
          >
            <ArrowLeft className="h-4 w-4 transition-transform group-hover:-translate-x-0.5" aria-hidden />
            Back
          </Link>

          <p className="mb-4 text-xs uppercase tracking-[0.25em] text-accent">
            {post.category}
          </p>
          <h1 className="font-display text-4xl font-bold leading-[1.05] tracking-tight md:text-6xl">
            {post.title}
          </h1>
          <p className="mt-6 text-sm tabular-nums text-ink-soft">
            {post.date} · {post.readingTime}
          </p>
        </motion.div>

        {post.image && (
          <motion.img
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.15, duration: 0.6 }}
            src={post.image}
            alt={`Illustration for “${post.title}”`}
            className="mt-10 aspect-[16/9] w-full rounded-sm object-cover"
          />
        )}

        <div className="mt-12 space-y-7">
          {post.body.map((para, i) => (
            <motion.p
              key={i}
              initial={{ opacity: 0, y: 14 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ duration: 0.45 }}
              className={`leading-relaxed ${
                i === 0 ? "text-lg text-ink" : "text-base text-ink/85"
              }`}
            >
              {para}
            </motion.p>
          ))}
        </div>

        <footer className="mt-16 border-t border-line pt-8">
          <Link
            to="/blog"
            className="inline-flex items-center gap-2 font-display text-lg font-semibold transition-colors hover:text-accent"
          >
            <ArrowLeft className="h-5 w-5" aria-hidden />
            More notes from the feed
          </Link>
        </footer>
      </article>
    </PageTransition>
  );
}
