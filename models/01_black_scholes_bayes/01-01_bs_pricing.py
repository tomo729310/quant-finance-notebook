import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import grad

def black_scholes_call(S, K, T, r, sigma):
    """BSモデルによるコール価格算出（JAX版）"""
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)

if __name__ == "__main__":
    # テスト用パラメータ
    s = 100.0  # 原資産価格
    k = 100.0  # 行使価格
    t = 1.0    # 残存期間
    r = 0.05   # 金利
    v = 0.2    # ボラティリティ

    # 1. 価格計算
    price = black_scholes_call(s, k, t, r, v)

    # 2. 自動微分によるデルタ(Delta)の計算
    # black_scholes_call の第1引数(s)で微分する関数を作成
    delta_func = grad(black_scholes_call, argnums=0)
    delta = delta_func(s, k, t, r, v)

    # 3. 自動微分によるベガ(Vega)の計算
    # 第5引数(v)で微分する関数を作成
    vega_func = grad(black_scholes_call, argnums=4)
    vega = vega_func(s, k, t, r, v)

    print("--- Black-Scholes Greeks Calculation ---")
    print(f"Option Price: {price:.4f}")
    print(f"Delta :    {delta:.4f}  <- 原資産の価格変化に対するオプション価格の変化額")
    print(f"Vega :    {vega:.4f}  <- 原資産のボラティリティ変化に対するオプション価格の変化額")