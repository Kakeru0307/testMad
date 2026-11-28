"""MAD を家のローカル GPU（RTX 等）で回す入口。Colab / Drive は使わない。

前提:
  - WAV はローカルに置く（既定: testMad/ref/wav と testMad/gen/wav）
  - CUDA 付き torch + mad_metric（requirements.txt）
  - 長時間かかるのでノートではなく本スクリプト推奨

使い方（testMad ルート、またはどこからでも可）:
  .\\venv\\Scripts\\python.exe local_mad.py
  .\\venv\\Scripts\\python.exe local_mad.py --batch-size 1
  .\\venv\\Scripts\\python.exe local_mad.py --clear-ref-embs
  .\\venv\\Scripts\\python.exe local_mad.py --ref-dir D:\\data\\ref_wav --gen-dir D:\\data\\gen_wav

スコアは -log(MAUVE)（低いほど参照に近い）。埋め込みは cache/ に残るので再実行は速い。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
DEFAULT_REF = ROOT / "ref" / "wav"
DEFAULT_GEN = ROOT / "gen" / "wav"
DEFAULT_REF_EMBS = ROOT / "cache" / "ref_embs"
DEFAULT_GEN_EMBS = ROOT / "cache" / "gen_embs"
DEFAULT_JSON = ROOT / "mad_result_local.json"


def _count_wavs(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".wav")


def _check_env() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "torch がありません。CUDA 版を入れてから:\n"
            f"  python -m pip install -r {ROOT / 'requirements.txt'}"
        ) from exc

    if not torch.cuda.is_available():
        print(
            "警告: CUDA が見えません。CPU だと極端に遅いです。"
            " RTX ドライバと CUDA 版 torch を確認してください。",
            file=sys.stderr,
        )
    else:
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"CUDA OK: {name}  VRAM≈{mem:.1f} GiB")

    try:
        import mad_metric  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "mad_metric がありません:\n"
            f"  python -m pip install -r {ROOT / 'requirements.txt'}"
        ) from exc

    try:
        import soundfile  # noqa: F401
    except ImportError:
        print(
            "警告: soundfile がありません。librosa が audioread に落ちることがあります。\n"
            "  python -m pip install soundfile",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local GPU MAD (MERT + MAUVE). No Colab / Drive."
    )
    parser.add_argument("--ref-dir", type=Path, default=DEFAULT_REF)
    parser.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN)
    parser.add_argument("--ref-embs-dir", type=Path, default=DEFAULT_REF_EMBS)
    parser.add_argument("--gen-embs-dir", type=Path, default=DEFAULT_GEN_EMBS)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_JSON,
        help=f"結果 JSON（既定 {DEFAULT_JSON.name}）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="VRAM 余裕があれば 2〜3。OOM なら 1 のまま",
    )
    parser.add_argument("--model-name", default="mert_330m")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--aggregation", default="max")
    parser.add_argument(
        "--clear-ref-embs",
        action="store_true",
        help="参照 WAV を差し替えたあと、ref 埋め込みキャッシュだけ消す",
    )
    parser.add_argument(
        "--clear-all-embs",
        action="store_true",
        help="ref/gen 両方の埋め込みキャッシュを消す",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="埋め込みをディスクに残さない（毎回フル再計算）",
    )
    args = parser.parse_args()

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))

    _check_env()

    ref_dir = args.ref_dir.resolve()
    gen_dir = args.gen_dir.resolve()
    if not ref_dir.is_dir():
        raise SystemExit(f"参照 WAV がありません: {ref_dir}")
    if not gen_dir.is_dir():
        raise SystemExit(f"生成 WAV がありません: {gen_dir}")

    n_ref = _count_wavs(ref_dir)
    n_gen = _count_wavs(gen_dir)
    if n_ref < 1 or n_gen < 1:
        raise SystemExit(f"WAV が足りません: n_ref={n_ref} n_gen={n_gen}")

    print(f"ref_dir={ref_dir}  n_ref={n_ref}")
    print(f"gen_dir={gen_dir}  n_gen={n_gen}")
    print(f"batch_size={args.batch_size}  model={args.model_name} layer={args.layer}")

    if args.clear_all_embs:
        for d in (args.ref_embs_dir, args.gen_embs_dir):
            if d.is_dir():
                shutil.rmtree(d)
                print(f"cleared {d}")
    elif args.clear_ref_embs and args.ref_embs_dir.is_dir():
        shutil.rmtree(args.ref_embs_dir)
        print(f"cleared {args.ref_embs_dir}")

    from compute_mad import compute_mad_score

    ref_embs = None if args.no_cache else args.ref_embs_dir
    gen_embs = None if args.no_cache else args.gen_embs_dir
    if ref_embs is not None:
        ref_embs.mkdir(parents=True, exist_ok=True)
    if gen_embs is not None:
        gen_embs.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    score = compute_mad_score(
        ref_dir,
        gen_dir,
        ref_embs_dir=ref_embs,
        gen_embs_dir=gen_embs,
        batch_size=int(args.batch_size),
        model_name=args.model_name,
        layer=int(args.layer),
        aggregation=args.aggregation,
    )
    elapsed = time.perf_counter() - t0

    result = {
        "mad": score,
        "polarity": "lower_better",
        "definition": "-log(MAUVE) via mad_metric (MERT-v1-330M)",
        "model_name": args.model_name,
        "layer": args.layer,
        "aggregation": args.aggregation,
        "batch_size": args.batch_size,
        "ref_dir": str(ref_dir),
        "gen_dir": str(gen_dir),
        "n_ref": n_ref,
        "n_gen": n_gen,
        "elapsed_sec": round(elapsed, 1),
        "runner": "local_mad.py",
    }
    line = (
        f"MAD={score:.6f}  (lower=better)  "
        f"n_ref={n_ref} n_gen={n_gen}  "
        f"elapsed={elapsed / 60.0:.1f} min"
    )
    print(line)

    out = args.json_out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
