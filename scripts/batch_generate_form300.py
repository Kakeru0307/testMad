"""MAD 用: generate_form で 300 本 MIDI を testMad/gen/midi に出力する。

各本は互いに異なる日本語テキスト（300 ユニーク）を使う。

使い方:
  ..\\venv\\Scripts\\python.exe testMad\\scripts\\batch_generate_form300.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PRTTYPE = REPO / "prttype"
sys.path.insert(0, str(PRTTYPE))

import torch

from generate_form import generate_form, _resolve_home_mode
from inference import load_model
from sample_structure_params import resolve_structure_params
from song_form import sample_song_form
from util.checkpoint_paths import resolve_part_checkpoint

OUT_DIR = Path(__file__).resolve().parents[1] / "gen" / "midi"
N = 300
MAX_ONSETS = 999
LEAD_SUSTAIN_TH = 0.75

# 組み合わせで 300 ユニーク文を作る（循環なし）
_MOODS = (
    "明るい",
    "楽しい",
    "元気な",
    "晴れやかな",
    "弾むような",
    "開放的な",
    "陽気な",
    "キラキラした",
    "しんみりした",
    "悲しい",
    "物悲しい",
    "寂しい",
    "沈んだ",
    "くすんだ",
    "諦め混じりの",
    "胸が痛む",
    "落ち着いた",
    "穏やかな",
    "ゆるやかな",
    "静かな",
    "ゆったりした",
    "やわらかい",
    "波のような",
    "緊迫した",
    "焦った",
    "張りつめた",
    "不穏な",
    "せわしない",
    "追い立てられる",
    "鋭い",
)
_SCENES = (
    "朝の高速道路",
    "夜の高速道路",
    "雨の部屋",
    "晴れの公園",
    "夕暮れの街",
    "深夜のスタジオ",
    "カフェの窓辺",
    "屋上の風",
    "地下ライブハウス",
    "海岸道路",
)
_TEXTURES = (
    "ギターバッキング",
    "ストローク伴奏",
    "パワーコード進行",
    "アルペジオ伴奏",
    "バンドアンサンブル",
    "ドライブギター",
    "ミュート主体の伴奏",
    "サビ前のリフ伴奏",
    "バラード伴奏",
    "ロックバッキング",
)


def build_unique_texts(n: int = N) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for mood in _MOODS:
        for scene in _SCENES:
            for texture in _TEXTURES:
                t = f"{mood}{scene}の{texture}"
                if t in seen:
                    continue
                seen.add(t)
                texts.append(t)
                if len(texts) >= n:
                    return texts
    raise RuntimeError(f"unique texts不足: got {len(texts)} need {n}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = build_unique_texts(N)
    assert len(prompts) == N
    assert len(set(prompts)) == N

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print(f"unique_texts={len(prompts)}", flush=True)
    models = {
        "device": device,
        "backing": load_model(resolve_part_checkpoint("backing"), device),
        "lead": load_model(resolve_part_checkpoint("lead"), device),
        "drum": load_model(resolve_part_checkpoint("drum"), device),
        "bass": load_model(resolve_part_checkpoint("bass"), device),
    }
    print("models loaded", flush=True)

    # テキスト一覧も残す（MAD 再現用）
    list_path = OUT_DIR.parent / "batch300_texts.txt"
    list_path.write_text("\n".join(f"{i:03d}\t{t}" for i, t in enumerate(prompts)), encoding="utf-8")
    print(f"texts -> {list_path}", flush=True)

    t0 = time.time()
    for i, text in enumerate(prompts):
        seed = i
        params = resolve_structure_params(text=text, seed=seed)
        home_mode = _resolve_home_mode(params.progression, params.mode or None)
        form = sample_song_form(
            seed=seed,
            home_progression=params.progression,
            home_mode=home_mode,
            va=params.va,
            bpm=params.bpm,
            song_energy=params.energy,
        )
        out = OUT_DIR / f"form_{i:03d}.mid"
        print(
            f"[{i:03d}/{N}] text={text!r} energy={params.energy} "
            f"tmpl={form.template_id} bars={form.total_bars} bpm={params.bpm} "
            f"-> {out.name}",
            flush=True,
        )
        generate_form(
            progression=params.progression,
            key=params.key,
            bpm=params.bpm,
            bars_per_chord=params.bars_per_chord,
            form=form,
            home_mode=home_mode,
            text=text,
            va=params.va,
            output=out,
            seed=seed,
            max_onsets_per_bar=MAX_ONSETS,
            lead_sustain_th=LEAD_SUSTAIN_TH,
            models=models,
        )
        elapsed = (time.time() - t0) / 60.0
        done = i + 1
        eta = elapsed / done * (N - done)
        print(f"  done {done}/{N} elapsed={elapsed:.1f}m eta={eta:.1f}m", flush=True)

    print(f"all done in {(time.time() - t0) / 60.0:.1f}m", flush=True)


if __name__ == "__main__":
    main()
