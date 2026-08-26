import { motion, useReducedMotion } from "framer-motion";
import type { TrendItem } from "../../data/trends";

type Props = {
  item: TrendItem;
};

export default function TrendTile({ item }: Props) {
  const reduce = useReducedMotion();

  return (
    <motion.a
      href="#trends"
      onClick={(e) => e.preventDefault()}
      aria-label={`${item.title} — ${item.category}`}
      whileHover={reduce ? undefined : { scale: 1.04, rotate: item.tilt }}
      transition={{ type: "spring", stiffness: 260, damping: 20 }}
      className="group relative block shrink-0 overflow-hidden rounded-sm bg-paper-deep"
      style={{ width: item.width, height: item.height, transformOrigin: "center" }}
    >
      <img
        src={item.src}
        alt={`${item.category}: ${item.title}`}
        loading="lazy"
        draggable={false}
        className="h-full w-full select-none object-cover transition-[filter] duration-500 group-hover:brightness-90"
      />
      {/* caption reveal */}
      <div
        className="pointer-events-none absolute inset-x-0 bottom-0 translate-y-2 bg-gradient-to-t from-ink/80 to-transparent p-4 opacity-0 transition-all duration-300 group-hover:translate-y-0 group-hover:opacity-100 group-focus-visible:translate-y-0 group-focus-visible:opacity-100"
      >
        <p className="text-xs uppercase tracking-[0.2em] text-white/70">
          {item.category}
          {item.author ? ` · @${item.author}` : ""}
        </p>
        <p className="font-display text-lg font-semibold leading-snug text-white">
          {item.title}
        </p>
      </div>
      <span className="sr-only">{`${item.category} — ${item.title}`}</span>
      <span
        aria-hidden
        className="absolute left-3 top-3 max-w-[70%] truncate rounded-full bg-paper/85 px-3 py-1 text-[11px] uppercase tracking-widest text-ink opacity-100 md:opacity-0 md:transition-opacity md:duration-300 md:group-hover:opacity-100"
      >
        {item.title}
      </span>
    </motion.a>
  );
}
