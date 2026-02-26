import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.scipy.stats import norm
import matplotlib.pyplot as plt

# 単一の価格計算関数
def bs_call_price(S, K, T, r, sigma):
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)

# --- JAXの魔法: vmap ---
# sigma（5番目の引数、インデックス4）だけが配列で、他は単一数値（None）として扱う
vmap_bs_price = vmap(bs_call_price, in_axes=(None, None, None, None, 0))
# 高速化のためにコンパイル
fast_vmap_bs = jit(vmap_bs_price)

if __name__ == "__main__":
    # ボラティリティを 5% から 100% まで 100刻みで作成
    vol_space = jnp.linspace(0.05, 1.0, 100)
    
    # 100個の価格を一瞬で計算
    prices = fast_vmap_bs(100.0, 100.0, 1.0, 0.05, vol_space)
    
    print(f"計算完了: 最初の5件の価格 = {prices[:5]}")
    
    # グラフ化
    # (環境によって headless の場合は print だけでOKです)
    plt.plot(vol_space, prices)
    plt.xlabel("Volatility (sigma)")
    plt.ylabel("Option Price")
    plt.title("Price Sensitivity to Volatility")
    plt.grid(True)
    plt.show()