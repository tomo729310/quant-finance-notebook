# Quant Finance Notebook

デリバティブ取引のための金融工学モデル・シミュレーションを  
Pythonで実装しながら学んでいくための個人ノートブック集

---

## 目的

- 金融工学・確率モデル・数値解析を 実装ベースで理解  
- NumPy / NumPyro / PyMC / JAX などの 確率的プログラミングツール の適用  
- 将来的に、モデル間の比較・リスク分析・シミュレーションへ発展  

---

## 構成

| ディレクトリ | 内容 |
|---------------|------|
| `01_black_scholes_bayes/` | ブラック・ショールズモデルのボラティリティをベイズ推定（NumPyroを取り入れてみる） |
| `02_yield_curve_modeling/` | イールドカーブのフィッティング（予定） |
| `03_monte_carlo_pricing/` | モンテカルロ法によるオプション価格推定（予定） |

---

## ⚙️ 環境構築

```bash
conda env create -f environment.yml
conda activate quant-finance
