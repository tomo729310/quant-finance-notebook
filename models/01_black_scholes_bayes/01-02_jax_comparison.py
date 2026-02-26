import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, grad, jit
from jax.scipy.stats import norm

# --- 1. ブラックショールズの定義 ---
def bs_call_price(S, K, T, r, sigma):
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)

# --- 2. JAXの機能 ---
# A. 自動微分 (Greeks計算に適用)
delta_fn = jit(grad(bs_call_price, argnums=0))

# B. 並列化 (1万個のデータをまとめて計算)
# vmapを使うと、単体用の関数を「配列用」に自動変換できる
vmap_bs_call = jit(vmap(bs_call_price, in_axes=(0, None, None, None, 0)))

# --- 3. 性能比較 ---
if __name__ == "__main__":
    n_samples = 10000
    spots = jnp.linspace(80, 120, n_samples)
    volatilities = jnp.linspace(0.1, 0.4, n_samples)
    
    print(f"【比較】{n_samples}件のオプション価格とデルタを計算します\n")

    # --- JAX版 (並列 + 高速化) ---
    # 初回実行（コンパイル）
    _ = vmap_bs_call(spots, 100.0, 1.0, 0.05, volatilities)
    
    start = time.time()
    prices_jax = vmap_bs_call(spots, 100.0, 1.0, 0.05, volatilities)
    jax_time = time.time() - start
    print(f"JAX版の実行時間: {jax_time:.6f} 秒")

    # --- 標準Python/NumPy版 (ループ処理) ---
    # 比較のために1件ずつループで回す（一般的な書き方）
    start = time.time()
    prices_py = []
    for i in range(n_samples):
        # 実際にはもっと複雑な処理になることが多い
        p = bs_call_price(spots[i], 100.0, 1.0, 0.05, volatilities[i])
        prices_py.append(p)
    py_time = time.time() - start
    print(f"Pythonループの実行時間: {py_time:.6f} 秒")

    print(f"\n JAXは約 {py_time/jax_time:.1f} 倍速い")
    
    # 自動微分の例も表示
    sample_delta = delta_fn(100.0, 100.0, 1.0, 0.05, 0.2)
    print(f"デルタ(自動微分)の計算結果: {sample_delta:.4f}")