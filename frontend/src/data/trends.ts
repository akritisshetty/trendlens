import type { LiveTile } from "../services/liveTiles";

export type TrendItem = {
  id: string;
  title: string;
  category: string;
  src: string;
  width: number;
  height: number;
  tilt: number;
  author?: string;
};

export type TrendRow = {
  label: string;
  direction: "ltr" | "rtl";
  duration: number;
  items: TrendItem[];
};

const img = (
  id: string,
  title: string,
  category: string,
  w: number,
  h: number,
  tilt: number
): TrendItem => ({
  id,
  title,
  category,
  src: `https://picsum.photos/seed/trendlens-${id}/${w * 2}/${h * 2}`,
  width: w,
  height: h,
  tilt,
});

export const trendRows: TrendRow[] = [
  {
    label: "food & drink",
    direction: "ltr",
    duration: 70,
    items: [
      img("f1", "Minimalist latte art", "Food", 300, 380, -1.2),
      img("f2", "Rustic brunch spread", "Food", 420, 320, 0.8),
      img("f3", "Sourdough crumb shots", "Food", 260, 340, 1.5),
      img("f4", "Dark moody plating", "Food", 360, 440, -0.6),
      img("f5", "Street food close-ups", "Food", 300, 300, 1),
      img("f6", "Pastel dessert tables", "Food", 340, 400, -1),
    ],
  },
  {
    label: "style & culture",
    direction: "rtl",
    duration: 85,
    items: [
      img("s1", "Scandinavian layering", "Fashion", 280, 400, 1.3),
      img("s2", "Thrifted vintage denim", "Fashion", 400, 320, -0.9),
      img("s3", "Gallery hop aesthetics", "Art", 300, 380, 0.7),
      img("s4", "Analog film grain looks", "Photography", 340, 300, -1.4),
      img("s5", "Brutalist architecture walks", "Design", 280, 420, 0.9),
      img("s6", "Night market neon", "Travel", 380, 340, -0.8),
    ],
  },
  {
    label: "sound & screens",
    direction: "ltr",
    duration: 95,
    items: [
      img("m1", "Bedroom pop visuals", "Music", 320, 360, -1.1),
      img("m2", "Vinyl corner styling", "Music", 280, 300, 0.9),
      img("m3", "Retro tech flat lays", "Technology", 400, 330, -0.7),
      img("m4", "Lo-fi desk setups", "Design", 300, 400, 1.2),
      img("m5", "Festival film photography", "Culture", 340, 320, -1),
      img("m6", "Cozy reading nooks", "Books", 280, 380, 0.8),
    ],
  },
];

/* ── Real Instagram tiles → marquee rows ──────────────────────── */

const TILE_SIZES: Array<[number, number, number]> = [
  // [width, height, tilt]
  [300, 380, -1.2],
  [420, 320, 0.8],
  [270, 350, 1.5],
  [360, 440, -0.6],
  [310, 310, 1],
  [340, 400, -1],
  [290, 410, 0.9],
  [400, 330, -1.4],
];

/** Build marquee rows from real backend tiles (round-robin over 3 rows). */
export function buildRowsFromLiveTiles(tiles: LiveTile[]): TrendRow[] {
  const rowsMeta = [
    { label: "live · row one", direction: "ltr" as const, duration: 70 },
    { label: "live · row two", direction: "rtl" as const, duration: 85 },
    { label: "live · row three", direction: "ltr" as const, duration: 95 },
  ];

  return rowsMeta.map((meta, rowIndex) => ({
    ...meta,
    items: tiles
      .filter((_, i) => i % rowsMeta.length === rowIndex)
      .map((tile, i) => {
        const [w, h, tilt] = TILE_SIZES[(i + rowIndex) % TILE_SIZES.length];
        return {
          id: tile.id,
          title: tile.title,
          category: tile.category,
          src: tile.url,
          width: w,
          height: h,
          tilt,
          author: tile.author,
        };
      }),
  }));
}
