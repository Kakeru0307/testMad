"""MAD（Mauve Audio Divergence）を計算する。testMad 単独用。

公式 mad_metric（MERT-v1-330M + MAUVE）を使う。
返り値は -log(MAUVE) なので **低いほど参照に近い**。

使い方（testMad ルートから）:
  .\\venv\\Scripts\\python.exe scripts\\compute_mad.py ref\\wav gen\\wav

1曲点数は出さず、フォルダ同士の距離を1つ出す。GPU 推奨（Colab 可）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _count_wavs(directory: Path) -> int:
    return sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".wav")


def compute_mad_score(
    ref_dir: Path,
    gen_dir: Path,
    *,
    ref_embs_dir: Path | None = None,
    gen_embs_dir: Path | None = None,
    batch_size: int = 3,
    model_name: str = "mert_330m",
    layer: int = 24,
    aggregation: str = "max",
) -> float:
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"参照ディレクトリがありません: {ref_dir}")
    if not gen_dir.is_dir():
        raise FileNotFoundError(f"生成ディレクトリがありません: {gen_dir}")

    n_ref = _count_wavs(ref_dir)
    n_gen = _count_wavs(gen_dir)
    if n_ref < 1:
        raise SystemExit(f"参照に WAV がありません: {ref_dir}")
    if n_gen < 1:
        raise SystemExit(f"生成に WAV がありません: {gen_dir}")

    try:
        from mad_metric import compute_mad
    except ImportError as exc:
        raise SystemExit(
            "mad_metric がありません。testMad\\venv で:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    # mad_metric API: eval_dir=生成, ref_dir=参照 → -log(MAUVE)
    score = float(
        compute_mad(
            eval_dir=str(gen_dir.resolve()),
            ref_dir=str(ref_dir.resolve()),
            eval_embs_dir=str(gen_embs_dir.resolve()) if gen_embs_dir else None,
            ref_embs_dir=str(ref_embs_dir.resolve()) if ref_embs_dir else None,
            batch_size=batch_size,
            model_name=model_name,
            layer=layer,
            aggregation=aggregation,
        )
    )
    return score


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="MAD (MERT + MAUVE) between two WAV folders [testMad]"
    )
    parser.add_argument("ref_dir", type=Path, help="参照 WAV ディレクトリ")
    parser.add_argument("gen_dir", type=Path, help="生成 WAV ディレクトリ")
    parser.add_argument(
        "--ref-embs-dir",
        type=Path,
        default=root / "cache" / "ref_embs",
        help="参照埋め込みキャッシュ（省略時 cache/ref_embs）",
    )
    parser.add_argument(
        "--gen-embs-dir",
        type=Path,
        default=root / "cache" / "gen_embs",
        help="生成埋め込みキャッシュ（省略時 cache/gen_embs）",
    )
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--model-name", default="mert_330m")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--aggregation", default="max")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="埋め込みをディスクに残さない",
    )
    args = parser.parse_args()

    ref_embs = None if args.no_cache else args.ref_embs_dir
    gen_embs = None if args.no_cache else args.gen_embs_dir
    if ref_embs is not None:
        ref_embs.mkdir(parents=True, exist_ok=True)
    if gen_embs is not None:
        gen_embs.mkdir(parents=True, exist_ok=True)

    score = compute_mad_score(
        args.ref_dir,
        args.gen_dir,
        ref_embs_dir=ref_embs,
        gen_embs_dir=gen_embs,
        batch_size=args.batch_size,
        model_name=args.model_name,
        layer=args.layer,
        aggregation=args.aggregation,
    )
    result = {
        "mad": score,
        "polarity": "lower_better",
        "definition": "-log(MAUVE) via mad_metric (MERT-v1-330M)",
        "model_name": args.model_name,
        "layer": args.layer,
        "aggregation": args.aggregation,
        "ref_dir": str(args.ref_dir.resolve()),
        "gen_dir": str(args.gen_dir.resolve()),
        "n_ref": _count_wavs(args.ref_dir),
        "n_gen": _count_wavs(args.gen_dir),
    }
    line = (
        f"MAD={score:.6f}  (lower=better)  "
        f"model={args.model_name} layer={args.layer} agg={args.aggregation}  "
        f"n_ref={result['n_ref']}  n_gen={result['n_gen']}"
    )
    print(line)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
