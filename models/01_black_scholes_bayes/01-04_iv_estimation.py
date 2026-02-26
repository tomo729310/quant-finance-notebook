import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import arviz as az

# BS式
def bs_call_price(S, K, T, r, sigma):
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)

# NumPyroモデル
def model(market_price):
    # 1. 事前分布: ボラティリティは 0%〜200% の間だろうと予測
    sigma = numpyro.sample("sigma", dist.Uniform(0.0, 2.0))
    
    # 2. 理論価格の計算
    theoretical_price = bs_call_price(100.0, 100.0, 1.0, 0.05, sigma)
    
    # 3. 観測モデル: 市場価格は理論価格の周りに少しの誤差(0.01)を伴って存在すると仮定
    numpyro.sample("obs", dist.Normal(theoretical_price, 0.01), obs=market_price)

if __name__ == "__main__":
    # 市場価格=12.5 を仮定
    target_market_price = 12.5
    
    # MCMCの設定
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=500)
    
    # 実行
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, market_price=target_market_price)
    
    # 結果の表示
    mcmc.print_summary()

# ArviZ形式に変換
    # ※NumPyroの結果を直接渡すだけで、詳細な統計解析が可能になります
    data = az.from_numpyro(mcmc)
    
    print("\nTrace Plotを表示")
    
    # A. トレースプロット（左：事後分布の密度、右：サンプリングの軌跡）
    az.plot_trace(data)
    plt.tight_layout()
    plt.show()
    
    # B. 事後分布の詳細（平均値やHDI: 最高密度区間を表示）
    az.plot_posterior(data)
    plt.title(f"Posterior Distribution of Implied Volatility (Market Price={target_market_price})")
    plt.show()