# testMad — MAD / FAD 参照評価（単独）

このディレクトリだけで MAD（主）と参照 WAV 準備が動く。親リポ `prttype/` に依存しない。

## 構成

| パス | 内容 |
|---|---|
| `scripts/compute_mad.py` | MAD（MERT + MAUVE、`-log(MAUVE)`、**低いほど良い**） |
| `scripts/midi_to_wav.py` | MIDI→WAV（FluidSynth + ローカル SF） |
| `assets/soundfonts/` | SF2 置き場（`default.sf2` は git 外） |
| `ref/midi/` | 参照 MIDI（チャートインスト） |
| `ref/wav/` | 参照 WAV（**git 外**。Drive に上げて Colab で使う） |
| `gen/midi/` | 本線 `generate_form.py` の生成 MIDI |
| `gen/wav/` | 生成側 WAV（**git 外**。Drive に上げて Colab で使う） |
| `requirements.txt` | 依存（venv をここに作る） |
| `colab_mad.ipynb` | Colab GPU 実行用 |

基準ドキュメント（研究リポ）: `docs/その他/MAD参照MIDI_採用基準と選曲一覧.md`

## セットアップ

```powershell
cd C:\Users\kake0\github\研究\testMad
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -U pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

FluidSynth が PATH にあること（MIDI→WAV 用）。

SoundFont はリポに含めない（サイズ制限）。`assets/soundfonts/default.sf2` に Fluid R3 GM（または同等の GM SF2）を置くか、環境変数 `SOUNDFONT` を設定する。

WAV（`ref/wav` / `gen/wav`）は git 対象外。ローカルで作った WAV を **Google Drive** に置き、`colab_mad.ipynb` から読む。再生成が要るときだけ `scripts/midi_to_wav.py` を使う。

## MIDI → WAV

```powershell
.\venv\Scripts\python.exe scripts\midi_to_wav.py ref\midi --out-dir ref\wav
```

## MAD

```powershell
.\venv\Scripts\python.exe scripts\compute_mad.py ref\wav gen\wav --json-out mad_result.json
```

本番の重い埋め込みは **Colab GPU**（`colab_mad.ipynb`）。ノートがリポを `git clone` し、WAV は Drive の `testMad/ref/wav`・`gen/wav` を読む。依存は `pip install mad_metric` のみ。
