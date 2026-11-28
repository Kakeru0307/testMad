"""一次仕分け: Demucs 歌ステム比で ref/wav を reject / uncertain / keep。

ユーザー許可（歌検出一次仕分け）。最終採用は目視。
閾値は listen_sample 20本で仮決め。

  .\\venv\\Scripts\\python.exe testMad\\scripts\\triage_ref_vocals.py
  .\\venv\\Scripts\\python.exe testMad\\scripts\\triage_ref_vocals.py --apply
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import librosa
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

ROOT = Path(__file__).resolve().parents[1]
WAV_DIR = ROOT / "ref" / "wav"
OUT_TSV = ROOT / "ref" / "manifests" / "vocal_triage.tsv"
REJECT_DIR = ROOT / "ref" / "wav_reject_vocal"
UNCERTAIN_DIR = ROOT / "ref" / "wav_uncertain_vocal"

# REJECT if any; KEEP if all keep-conds; else UNCERTAIN
REJECT_RMS = 0.12
REJECT_P90 = 0.40
REJECT_FRAC = 0.25
KEEP_RMS = 0.05
KEEP_P90 = 0.15
KEEP_FRAC = 0.10
CROP_SEC = 15.0


def classify(rms: float, p90: float, frac: float) -> str:
    if rms >= REJECT_RMS or p90 >= REJECT_P90 or frac >= REJECT_FRAC:
        return "reject"
    if rms < KEEP_RMS and p90 < KEEP_P90 and frac < KEEP_FRAC:
        return "keep"
    return "uncertain"


def vocal_metrics(model, path: Path) -> tuple[float, float, float, float, float]:
    y, _sr = librosa.load(path, sr=model.samplerate, mono=False)
    if y.ndim == 1:
        wav = torch.from_numpy(y).float()[None].repeat(2, 1)
    else:
        wav = torch.from_numpy(y).float()
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
    n = int(CROP_SEC * model.samplerate)
    if wav.shape[-1] > n:
        s = (wav.shape[-1] - n) // 2
        wav = wav[:, s : s + n]
    t0 = time.time()
    with torch.no_grad():
        out = apply_model(
            model, wav[None], device="cpu", split=True, overlap=0.1, progress=False
        )
    dt = time.time() - t0
    vi = list(model.sources).index("vocals")
    voc = out[0, vi]
    acc = out[0, [i for i, s in enumerate(model.sources) if s != "vocals"]].sum(0)
    mix_e = float(wav.pow(2).mean().sqrt() + 1e-8)
    voc_e = float(voc.pow(2).mean().sqrt())
    rms = voc_e / mix_e
    frame = model.samplerate // 4
    v2 = voc.pow(2).mean(0)
    a2 = acc.pow(2).mean(0)
    n = min(v2.numel(), a2.numel())
    nf = max(1, n // frame)
    v2 = v2[: nf * frame].view(nf, frame).mean(1)
    a2 = a2[: nf * frame].view(nf, frame).mean(1)
    ratio = v2.sqrt() / (a2.sqrt() + 1e-8)
    p50 = float(ratio.median())
    p90 = float(ratio.kthvalue(max(1, int(0.9 * nf))).values)
    frac = float((ratio > 0.5).float().mean())
    return dt, rms, p50, p90, frac


def load_done(tsv: Path) -> dict[str, dict[str, str]]:
    if not tsv.is_file():
        return {}
    done: dict[str, dict[str, str]] = {}
    with tsv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            done[row["file"]] = row
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="reject/uncertain を wav から移動（triage TSV 必須）",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)

    if args.apply:
        done = load_done(OUT_TSV)
        if not done:
            raise SystemExit(f"no triage yet: {OUT_TSV}")
        REJECT_DIR.mkdir(parents=True, exist_ok=True)
        UNCERTAIN_DIR.mkdir(parents=True, exist_ok=True)
        moved = {"reject": 0, "uncertain": 0, "keep": 0}
        for name, row in done.items():
            src = WAV_DIR / name
            if not src.is_file():
                continue
            label = row["label"]
            if label == "reject":
                src.rename(REJECT_DIR / name)
                moved["reject"] += 1
            elif label == "uncertain":
                src.rename(UNCERTAIN_DIR / name)
                moved["uncertain"] += 1
            else:
                moved["keep"] += 1
        print("moved", moved, "keep_left_in_wav", moved["keep"])
        return

    wavs = sorted(WAV_DIR.glob("*.wav"))
    if args.limit > 0:
        wavs = wavs[: args.limit]
    done = load_done(OUT_TSV)
    todo = [p for p in wavs if p.name not in done]
    print(f"total={len(wavs)} done={len(done)} todo={len(todo)} out={OUT_TSV}")

    write_header = not OUT_TSV.is_file()
    model = get_model("htdemucs")
    model.eval()

    with OUT_TSV.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["file", "label", "rms", "p50", "p90", "frac", "seconds"],
            delimiter="\t",
        )
        if write_header:
            w.writeheader()
            f.flush()
        for i, path in enumerate(todo, 1):
            try:
                dt, rms, p50, p90, frac = vocal_metrics(model, path)
                label = classify(rms, p90, frac)
                row = {
                    "file": path.name,
                    "label": label,
                    "rms": f"{rms:.6f}",
                    "p50": f"{p50:.6f}",
                    "p90": f"{p90:.6f}",
                    "frac": f"{frac:.4f}",
                    "seconds": f"{dt:.2f}",
                }
            except Exception as exc:  # noqa: BLE001
                row = {
                    "file": path.name,
                    "label": "uncertain",
                    "rms": "",
                    "p50": "",
                    "p90": "",
                    "frac": "",
                    "seconds": f"error:{exc}",
                }
                label = "uncertain"
            w.writerow(row)
            f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"progress {i}/{len(todo)} last={path.name} -> {label}", flush=True)

    # summary
    all_rows = load_done(OUT_TSV)
    from collections import Counter

    c = Counter(r["label"] for r in all_rows.values())
    print("summary", dict(c), "n", len(all_rows))


if __name__ == "__main__":
    main()
