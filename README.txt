# Quant Finance Notebook

デリバティブ取引のための金融工学モデル・シミュレーションを  
Pythonで実装しながら学んでいくための個人ノートブック集  

---

## 目的

- 金融工学・確率モデル・数値解析を 実装ベースで理解  
- NumPy / NumPyro / PyMC / JAX などの 確率的プログラミングツール の適用  
- 将来的に、モデル間の比較・リスク分析・シミュレーションへ発展  
- コードは0から自作せず、色々参考にしながら作成

---

## 構成

| ディレクトリ | 内容 |
|---------------|------|
| `01_black_scholes_bayes/` | ブラック・ショールズモデルのボラティリティをベイズ推定（NumPyroを取り入れてみる） |
| `02_yield_curve_modeling/` | イールドカーブのフィッティング（予定） |
| `03_monte_carlo_pricing/` | モンテカルロ法によるオプション価格推定（予定） |

---

## 環境構築

Pythonのパッケージ管理ツール 'uv'を使用
```bash
   uv sync
```
## セットアップ (Apple Silicon / uv)

Apple Silicon環境でjaxを動作させるための手順。

### 1. 仮想環境の構築
Conda が有効な場合は、干渉を避けるため一度無効化してから実行する。

```bash
# Conda環境の無効化（(base)が表示されている場合）
conda deactivate

# arm64版Pythonを指定して環境構築
uv python install 3.11-arm64
uv sync --python 3.11-arm64

### 2. 実行方法
```bash
# 1. 最初に一度だけ環境を固定
uv venv --python 3.11-arm64

# 2. 実行（uv run がこの環境を優先して使用）
uv run models/01_black_scholes_bayes/bs_pricing.py
```

