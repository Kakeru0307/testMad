"""MAD gen: form_000..299 を文→prior→song_form で生成（model cache）。

使い方（リポジトリルート）:
  .\\venv\\Scripts\\python.exe testMad\\scripts\\batch_gen_form300.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRTTTYPE = ROOT / "prttype"
sys.path.insert(0, str(PRTTTYPE))

from generate_backing import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_PRIOR_CHECKPOINT,
)
from generate_bass import DEFAULT_CHECKPOINT as DEFAULT_BASS_CHECKPOINT  # noqa: E402
from generate_drum import DEFAULT_CHECKPOINT as DEFAULT_DRUM_CHECKPOINT  # noqa: E402
from generate_form import generate_form  # noqa: E402
from generate_lead import DEFAULT_CHECKPOINT as DEFAULT_LEAD_CHECKPOINT  # noqa: E402
from inference import load_model  # noqa: E402
from sample_structure_params import resolve_structure_params  # noqa: E402
from song_form import sample_song_form  # noqa: E402

OUT_MIDI = ROOT / "testMad" / "gen" / "midi"
N = 300
GROUPS = ("bright", "sad", "calm", "tense")


def _load_prompt_groups() -> dict[str, list[str]]:
    import importlib.util

    path = PRTTTYPE / "scripts" / "generate_prior_pair_candidates.py"
    spec = importlib.util.spec_from_file_location("prior_pair_cand", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod._PROMPT_GROUPS)


def _text_for_index(i: int, groups: dict[str, list[str]]) -> str:
    group = GROUPS[i % len(GROUPS)]
    prompts = groups[group]
    return prompts[(i // len(GROUPS)) % len(prompts)]


def main() -> None:
    import torch

    prompt_groups = _load_prompt_groups()
    OUT_MIDI.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    models = {
        "device": device,
        "backing": load_model(DEFAULT_CHECKPOINT, device),
        "lead": load_model(DEFAULT_LEAD_CHECKPOINT, device),
        "drum": load_model(DEFAULT_DRUM_CHECKPOINT, device),
        "bass": load_model(DEFAULT_BASS_CHECKPOINT, device),
    }
    print("models loaded")

    t0 = time.time()
    for i in range(N):
        text = _text_for_index(i, prompt_groups)
        seed = i
        out = OUT_MIDI / f"form_{i:03d}.mid"
        params = resolve_structure_params(
            text=text,
            prior_checkpoint=DEFAULT_PRIOR_CHECKPOINT,
            seed=seed,
            sample=True,
        )
        from generate_form import _resolve_home_mode

        home_mode = _resolve_home_mode(params.progression, params.mode or None)
        form = sample_song_form(
            seed=seed,
            home_progression=params.progression,
            home_mode=home_mode,
            va=params.va,
            bpm=params.bpm,
        )
        sec = form.total_bars * 240.0 / max(40.0, float(params.bpm))
        print(
            f"[{i:03d}/{N}] text={text!r} tmpl={form.template_id} "
            f"bars={form.total_bars} bpm={params.bpm:.0f} ~{sec:.0f}s -> {out.name}",
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
            checkpoint=DEFAULT_CHECKPOINT,
            lead_checkpoint=DEFAULT_LEAD_CHECKPOINT,
            bass_checkpoint=DEFAULT_BASS_CHECKPOINT,
            drum_checkpoint=DEFAULT_DRUM_CHECKPOINT,
            output=out,
            seed=seed,
            models=models,
        )
        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 1e-6)
        eta = (N - i - 1) / max(rate, 1e-6)
        print(f"  done {i+1}/{N} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)

    print(f"all done in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
