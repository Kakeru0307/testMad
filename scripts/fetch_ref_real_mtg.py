"""Fetch MTG instrumentalrock via Jamendo (primary) / freesound CDN (fallback).

Jamendo full-track MP3 is trimmed to 30s WAV. Avoids freesound 503 storms.
"""
from __future__ import annotations

import csv
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "ref_real"
META = REF / "meta" / "raw_30s_cleantags.tsv"
MANIFEST = REF / "manifests" / "mtg_instrumentalrock.tsv"
FAILED = REF / "manifests" / "failed_mtg.tsv"
URLS_TXT = REF / "manifests" / "mtg_urls_jamendo.txt"
MP3_DIR = REF / "mp3" / "mtg"
WAV_DIR = REF / "wav"
CDN = "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio"

VOICE_NEEDLES = ("voice", "vocal", "sing", "choir", "spoken")
SLEEP_S = 0.4
TIMEOUT_S = 90
RETRIES = 4


def jamendo_url(numeric_id: str) -> str:
    # yt-dlp / public app redirect pattern; returns full-track mp3
    return f"https://mp3d.jamendo.com/?trackid={numeric_id}&format=mp32&from=app-97dab294"


def build_manifest() -> list[dict]:
    rows: list[dict] = []
    with META.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("TRACK_ID"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            track_id, _artist, _album, path, duration = parts[:5]
            tags = parts[5:]
            if "genre---instrumentalrock" not in tags:
                continue
            if any(any(n in t.lower() for n in VOICE_NEEDLES) for t in tags):
                continue
            nid = track_id.replace("track_", "").lstrip("0") or "0"
            # keep zero-pad for filenames from full track_id digits
            tid_num = track_id.replace("track_", "")
            rows.append(
                {
                    "track_id": track_id,
                    "path": path,
                    "duration": duration,
                    "url_jamendo": jamendo_url(nid),
                    "url_cdn": f"{CDN}/{path}",
                    "tid_num": tid_num,
                }
            )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["track_id", "path", "duration", "url_jamendo", "url_cdn"],
            delimiter="\t",
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ["track_id", "path", "duration", "url_jamendo", "url_cdn"]})
    missing = [
        r
        for r in rows
        if not ((WAV_DIR / f"mtg_{r['tid_num']}.wav").exists() and (WAV_DIR / f"mtg_{r['tid_num']}.wav").stat().st_size > 1000)
    ]
    URLS_TXT.write_text("\n".join(r["url_jamendo"] for r in missing) + ("\n" if missing else ""), encoding="utf-8")
    print(f"jamendo url list ({len(missing)} remaining) -> {URLS_TXT}", flush=True)
    return rows


def download_one(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(RETRIES + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"},
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp, tmp.open("wb") as out:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    out.write(chunk)
            if tmp.stat().st_size < 1000:
                raise RuntimeError(f"too small: {tmp.stat().st_size}")
            tmp.replace(dest)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError, OSError) as e:
            last_err = e
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(str(last_err))


def to_wav(mp3: Path, wav: Path) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp3),
        "-t",
        "30",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(wav),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:] if r.stderr else "ffmpeg failed")


def process_rows(rows: list[dict], *, label: str) -> tuple[int, int, list[dict]]:
    failed: list[dict] = []
    ok = skip = 0
    for i, row in enumerate(rows, 1):
        stem = f"mtg_{row['tid_num']}"
        mp3 = MP3_DIR / f"{stem}.mp3"
        wav = WAV_DIR / f"{stem}.wav"
        if wav.exists() and wav.stat().st_size > 1000:
            skip += 1
            if i % 25 == 0 or i == len(rows):
                print(f"{label} {i}/{len(rows)} ok={ok} skip={skip} fail={len(failed)}", flush=True)
            continue
        try:
            if not (mp3.exists() and mp3.stat().st_size > 1000):
                try:
                    download_one(row["url_jamendo"], mp3)
                except Exception as e1:  # noqa: BLE001
                    print(f"  jamendo fail {row['track_id']}: {e1}; try CDN", flush=True)
                    download_one(row["url_cdn"], mp3)
                time.sleep(SLEEP_S)
            to_wav(mp3, wav)
            ok += 1
        except Exception as e:  # noqa: BLE001
            failed.append(
                {
                    "track_id": row["track_id"],
                    "path": row["path"],
                    "duration": row["duration"],
                    "url": row["url_jamendo"],
                    "error": str(e)[:300],
                }
            )
            print(f"FAIL {row['track_id']}: {e}", flush=True)
        if i % 25 == 0 or i == len(rows):
            print(f"{label} {i}/{len(rows)} ok={ok} skip={skip} fail={len(failed)}", flush=True)
    return ok, skip, failed


def main() -> int:
    if not META.exists():
        print(f"missing meta: {META}", file=sys.stderr)
        return 1
    rows = build_manifest()
    print(f"manifest {len(rows)} -> {MANIFEST}", flush=True)
    MP3_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    ok1, skip1, failed = process_rows(rows, label="progress")
    if failed:
        print(f"retry pass for {len(failed)} after 30s...", flush=True)
        time.sleep(30.0)
        # rebuild row dicts with tid_num for retry
        by_id = {r["track_id"]: r for r in rows}
        retry_rows = [by_id[f["track_id"]] for f in failed if f["track_id"] in by_id]
        ok2, skip2, failed2 = process_rows(retry_rows, label="retry")
        ok1 += ok2
        skip1 += skip2
        failed = failed2

    with FAILED.open("w", encoding="utf-8", newline="") as f:
        fields = ["track_id", "path", "duration", "url", "error"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(failed)
    print(f"done ok={ok1} skip={skip1} fail={len(failed)} failed_tsv={FAILED}", flush=True)
    return 0 if ok1 + skip1 > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
