"""MIDI → WAV（FluidSynth + SoundFont）。testMad 単独用。

使い方（testMad ルートから）:
  .\\venv\\Scripts\\python.exe scripts\\midi_to_wav.py gen\\midi --out-dir gen\\wav

既定: default.sf2（GM）のあと VDG_1.1.sf2 があれば重ね読み（program 29/30 を歪みギターに）。
環境変数:
  SOUNDFONT           … ベース GM
  SOUNDFONT_OVERLAY   … 上書き用（VDG 等）
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SF_DIR = ROOT / "assets" / "soundfonts"
DEFAULT_SF2_CANDIDATES = (
    Path(os.environ["SOUNDFONT"]) if os.environ.get("SOUNDFONT") else None,
    SF_DIR / "default.sf2",
    REPO / "prttype" / "assets" / "soundfonts" / "default.sf2",
    Path.home() / "soundfonts" / "default.sf2",
)
OVERLAY_SF2_CANDIDATES = (
    Path(os.environ["SOUNDFONT_OVERLAY"]) if os.environ.get("SOUNDFONT_OVERLAY") else None,
    SF_DIR / "VDG_1.1.sf2",
    SF_DIR / "rock_guitar.sf2",
    REPO / "prttype" / "assets" / "soundfonts" / "VDG_1.1.sf2",
    REPO / "prttype" / "assets" / "soundfonts" / "rock_guitar.sf2",
)


def find_fluidsynth() -> str:
    exe = shutil.which("fluidsynth")
    if exe:
        return exe
    raise FileNotFoundError(
        "fluidsynth が見つかりません。scoop install extras/fluidsynth などで入れてください。"
    )


def _first_existing(candidates: tuple[Path | None, ...]) -> Path | None:
    for p in candidates:
        if p is not None and p.is_file():
            return p
    return None


def resolve_soundfont(explicit: Path | None) -> Path:
    if explicit is not None and explicit.is_file():
        return explicit
    found = _first_existing(DEFAULT_SF2_CANDIDATES)
    if found is not None:
        return found
    raise FileNotFoundError(
        "SoundFont (.sf2) がありません。次のいずれかに置いてください:\n"
        f"  - {SF_DIR / 'default.sf2'}\n"
        f"  - {Path.home() / 'soundfonts' / 'default.sf2'}\n"
        "または環境変数 SOUNDFONT を設定。"
    )


def resolve_overlay(explicit: Path | None, *, enabled: bool) -> Path | None:
    if not enabled:
        return None
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    return _first_existing(OVERLAY_SF2_CANDIDATES)


def remap_lead_program(midi_path: Path, program: int) -> Path:
    """Lead トラックの program だけ差し替えた一時 MIDI を返す。"""
    import muspy

    music = muspy.read_midi(str(midi_path))
    changed = False
    for tr in music.tracks:
        if tr.is_drum:
            continue
        if (tr.name or "") == "Lead":
            if int(tr.program) != int(program):
                tr.program = int(program)
                changed = True
    if not changed:
        return midi_path
    tmp = Path(tempfile.mkstemp(prefix=midi_path.stem + "_", suffix=".mid")[1])
    muspy.write_midi(str(tmp), music)
    return tmp


def midi_to_wav(
    midi_path: Path,
    wav_path: Path,
    *,
    soundfont: Path,
    overlay: Path | None = None,
    sample_rate: int = 44100,
    lead_program: int | None = None,
) -> Path:
    fs = find_fluidsynth()
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    render_mid = midi_path
    tmp: Path | None = None
    if lead_program is not None:
        render_mid = remap_lead_program(midi_path, lead_program)
        if render_mid != midi_path:
            tmp = render_mid
    cmd = [
        fs,
        "-ni",
        "-F",
        str(wav_path.resolve()),
        "-r",
        str(sample_rate),
        str(soundfont.resolve()),
    ]
    if overlay is not None:
        cmd.append(str(overlay.resolve()))
    cmd.append(str(render_mid.resolve()))
    print("+", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    finally:
        if tmp is not None and tmp.is_file():
            # Windows: FluidSynth 解放直後は WinError 32 になり得る
            import time

            for i in range(6):
                try:
                    tmp.unlink(missing_ok=True)
                    break
                except PermissionError:
                    time.sleep(0.25 * (i + 1))
    return wav_path


def main() -> None:
    parser = argparse.ArgumentParser(description="MIDI → WAV (FluidSynth) [testMad]")
    parser.add_argument("input", type=Path, help="MIDI ファイルまたはディレクトリ")
    parser.add_argument("--out-dir", type=Path, default=None, help="出力ディレクトリ")
    parser.add_argument("--soundfont", type=Path, default=None)
    parser.add_argument(
        "--overlay",
        type=Path,
        default=None,
        help="後段 SF2（VDG 等）。省略時は VDG_1.1.sf2 を自動探索",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="重ね読みしない（GM のみ）",
    )
    parser.add_argument(
        "--lead-program",
        type=int,
        default=30,
        help="Lead の GM program（既定 30=Distortion。負で変更しない）",
    )
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()

    sf2 = resolve_soundfont(args.soundfont)
    overlay = resolve_overlay(args.overlay, enabled=not args.no_overlay)
    lead_prog = None if int(args.lead_program) < 0 else int(args.lead_program)
    print(f"base={sf2}")
    print(f"overlay={overlay}" if overlay else "overlay=(none)")
    if lead_prog is not None:
        print(f"lead_program={lead_prog}")

    inp = args.input
    if inp.is_file():
        midis = [inp]
    elif inp.is_dir():
        midis = sorted(inp.rglob("*.mid")) + sorted(inp.rglob("*.midi"))
    else:
        raise FileNotFoundError(inp)
    if not midis:
        raise SystemExit(f"MIDI がありません: {inp}")

    out_dir = args.out_dir
    for mid in midis:
        if out_dir is None:
            wav = mid.with_suffix(".wav")
        else:
            wav = out_dir / (mid.stem + ".wav")
        if wav.is_file() and wav.stat().st_size > 1000:
            print(f"skip exists {wav}")
            continue
        midi_to_wav(
            mid,
            wav,
            soundfont=sf2,
            overlay=overlay,
            sample_rate=args.sample_rate,
            lead_program=lead_prog,
        )
        print(f"wrote {wav}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
