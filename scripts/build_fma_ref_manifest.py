"""Build FMA rock ∩ instrumental (30s–6.5min) ID list from fma_metadata.zip."""
from __future__ import annotations

import ast
import csv
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "ref_real" / "meta" / "fma_metadata.zip"
OUT = ROOT / "ref_real" / "manifests" / "fma_rock_instrumental.tsv"

# Prior count: top_level Rock=12 ∩ Instrumental=1235, duration 30–390s → 1432
ROCK_TOP = 12
INST_TOP = 1235
DUR_MIN = 30.0
DUR_MAX = 6.5 * 60.0


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"missing {ZIP_PATH}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(ZIP_PATH) as z:
        genres: dict[int, dict] = {}
        with z.open("fma_metadata/genres.csv") as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                genres[int(row["genre_id"])] = {
                    "title": row["title"],
                    "top": int(row["top_level"]),
                }
        rock_ids = {gid for gid, g in genres.items() if g["top"] == ROCK_TOP}
        inst_ids = {gid for gid, g in genres.items() if g["top"] == INST_TOP}

        raw = z.open("fma_metadata/tracks.csv")
        text = io.TextIOWrapper(raw, encoding="utf-8")
        hg = next(csv.reader(text))
        hn = next(csv.reader(text))
        next(text)  # units row
        cols = list(zip(hg, hn))

        def idx(group: str, name: str) -> int:
            for i, (g, n) in enumerate(cols):
                if g == group and n == name:
                    return i
            raise KeyError(f"{group}/{name}")

        i_tid = 0  # track_id is first column / index name
        # tracks.csv: first column is track_id under ('', 'track_id') or similar
        for i, (g, n) in enumerate(cols):
            if n == "track_id":
                i_tid = i
                break
        i_dur = idx("track", "duration")
        i_genres = idx("track", "genres_all")
        i_subset = idx("set", "subset")

        rows_out: list[dict] = []
        reader = csv.reader(text)
        for row in reader:
            try:
                tid = int(row[i_tid])
                dur = float(row[i_dur])
                g_all = set(ast.literal_eval(row[i_genres]))
            except (ValueError, SyntaxError):
                continue
            if not (g_all & rock_ids and g_all & inst_ids):
                continue
            if not (DUR_MIN <= dur <= DUR_MAX):
                continue
            folder = f"{tid // 1000:03d}"
            fname = f"{tid:06d}.mp3"
            rel = f"fma_large/{folder}/{fname}"
            rows_out.append(
                {
                    "track_id": tid,
                    "duration": dur,
                    "subset": row[i_subset],
                    "zip_path": rel,
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "duration", "subset", "zip_path"], delimiter="\t")
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {len(rows_out)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
