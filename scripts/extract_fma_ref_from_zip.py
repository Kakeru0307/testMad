"""Extract FMA IDs from fma_large.zip, convert to WAV, then optionally delete zip.

Space-safe order: extract needed mp3 → convert → delete mp3 → (caller deletes zip).
"""
from __future__ import annotations

import csv
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "ref_real"
MANIFEST = REF / "manifests" / "fma_rock_instrumental.tsv"
FAILED = REF / "manifests" / "failed_fma.tsv"
ZIP_PATH = REF / "meta" / "fma_large.zip"
MP3_DIR = REF / "mp3" / "fma"
WAV_DIR = REF / "wav"


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


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"missing zip: {ZIP_PATH}", file=sys.stderr)
        return 1
    if not MANIFEST.exists():
        print(f"missing manifest: {MANIFEST}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    with MANIFEST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    MP3_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[dict] = []
    ok = skip = 0

    with zipfile.ZipFile(ZIP_PATH) as z:
        names = set(z.namelist())
        for i, row in enumerate(rows, 1):
            tid = int(row["track_id"])
            zip_path = row["zip_path"]
            # zip may store with or without fma_large/ prefix variants
            candidates = [zip_path, zip_path.replace("fma_large/", "")]
            member = next((c for c in candidates if c in names), None)
            if member is None:
                # try suffix match
                suffix = f"{tid:06d}.mp3"
                member = next((n for n in names if n.endswith(suffix)), None)
            stem = f"fma_{tid:06d}"
            mp3 = MP3_DIR / f"{stem}.mp3"
            wav = WAV_DIR / f"{stem}.wav"
            if wav.exists() and wav.stat().st_size > 1000:
                skip += 1
                continue
            try:
                if member is None:
                    raise FileNotFoundError(f"not in zip: {zip_path}")
                if not (mp3.exists() and mp3.stat().st_size > 1000):
                    with z.open(member) as src, mp3.open("wb") as dst:
                        while True:
                            chunk = src.read(1 << 16)
                            if not chunk:
                                break
                            dst.write(chunk)
                to_wav(mp3, wav)
                mp3.unlink(missing_ok=True)
                ok += 1
            except Exception as e:  # noqa: BLE001
                failed.append({**row, "error": str(e)[:300]})
                print(f"FAIL {tid}: {e}", flush=True)
            if i % 50 == 0 or i == len(rows):
                print(f"progress {i}/{len(rows)} ok={ok} skip={skip} fail={len(failed)}", flush=True)

    with FAILED.open("w", encoding="utf-8", newline="") as f:
        fields = list(rows[0].keys()) + ["error"] if rows else ["track_id", "error"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(failed)
    print(f"done ok={ok} skip={skip} fail={len(failed)}", flush=True)
    print("When satisfied, delete the zip to free ~93GB:", ZIP_PATH, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
