"""Download every Instagram post image that is not yet on disk.

Idempotent: posts whose local file already exists are skipped, so this is safe
to re-run and will also pick up posts that were merged into the corpus on an
earlier run whose image download never completed.

Posts are processed newest-first on purpose. Instagram's CDN URLs are signed
and expire within hours to a day, so the URLs captured for older posts during
an earlier fetch are long dead and 404. Walking the corpus in timestamp order
therefore spends most of the run re-requesting URLs that cannot ever succeed,
while the freshly fetched posts -- the only ones whose images are still
downloadable -- sit at the end of the queue.
"""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.data_collector import (  # noqa: E402
    _local_image_path,
    download_images,
)

BATCH = 100


def main() -> int:
    df = pd.read_parquet(config.INSTAGRAM_DIR / "all_posts.parquet")
    df = df[df["image_url"].notna() & (df["image_url"] != "")].copy()
    df["_ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("_ts", ascending=False)

    need = df[
        [not _local_image_path(r.post_id, r.image_url).exists() for r in df.itertuples()]
    ].drop(columns=["_ts"])

    print(f"corpus={len(df)} missing_image={len(need)}", flush=True)
    if need.empty:
        print("nothing to download", flush=True)
        return 0

    started = time.time()
    saved_total = 0
    for i in range(0, len(need), BATCH):
        chunk = need.iloc[i : i + BATCH]
        saved_total += len(download_images(chunk))
        done = min(i + BATCH, len(need))
        rate = saved_total / max(time.time() - started, 1e-9)
        print(
            f"  {done}/{len(need)} queued, {saved_total} saved "
            f"({rate:.1f}/s, {time.time() - started:.0f}s elapsed)",
            flush=True,
        )

    print(f"DOWNLOADED {saved_total} images in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())