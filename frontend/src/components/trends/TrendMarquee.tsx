import { Fragment } from "react";
import { useReducedMotion } from "framer-motion";
import type { TrendRow } from "../../data/trends";
import TrendTile from "./TrendTile";

type Props = {
  row: TrendRow;
};

/**
 * Infinite horizontal marquee. Content is duplicated once so the -50%
 * keyframe loops seamlessly. Rows alternate direction; hover/focus pauses.
 */
export default function TrendMarqueeRow({ row }: Props) {
  const reduce = useReducedMotion();
  const doubled = [...row.items, ...row.items];

  return (
    <div
      className="marquee-row relative overflow-hidden py-3"
      role="list"
      aria-label={`Trending in ${row.label}`}
    >
      <div
        className="marquee-track"
        data-direction={
          reduce ? undefined : row.direction === "rtl" ? "right" : "left"
        }
        style={{ ["--marquee-duration" as string]: `${row.duration}s` }}
      >
        {doubled.map((item, i) => (
          <Fragment key={`${item.id}-${i}`}>
            <div role="listitem" className="px-2 md:px-3">
              <TrendTile item={item} />
            </div>
          </Fragment>
        ))}
      </div>
      {/* edge fades */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-y-0 left-0 w-16 bg-gradient-to-r from-paper to-transparent md:w-32"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-y-0 right-0 w-16 bg-gradient-to-l from-paper to-transparent md:w-32"
      />
    </div>
  );
}
