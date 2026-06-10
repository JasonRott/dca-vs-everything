"""上帝視角純網格基準回測。

設計（理論最佳解的上限基準，刻意使用不可得的未來資訊）：
- 每期 = 日曆月（UTC）。期初即知道本月的最高價 H 與最低價 L → 設為等比網格上下限。
- 格數由本月「真實」日波動率決定：間距 d = k × σ_daily，
  n = round(ln(H/L) / ln(1+d))，k 為敏感度參數。
- 期初以第一棒開盤價啟動網格；月底最後一棒收盤價市價清算全部持幣，
  淨值滾入下月新網格。
- 對照基準：BTC 買入持有。

輸出：results/god_view_monthly_k{K}.csv、results/god_view_summary.csv、
      results/god_view_equity.png、results/god_view_diagnostics.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_binance import load_klines
from grid_engine import GeometricGridBot

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp01_god_view"
RESULTS.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
START, END = "2018-01-01", "2026-06-01"
FEE = 0.001
K_LIST = [0.5, 1.0, 2.0, 3.0]
INIT_CAPITAL = 10_000.0
MAX_GRIDS = 500


def month_stats(bars: pd.DataFrame) -> dict:
    """單月的上帝視角參數：高低價與日波動率。"""
    daily_close = bars["close"].resample("1D").last().dropna()
    logret = np.log(daily_close).diff().dropna()
    vol_d = float(logret.std())
    return {
        "low": float(bars["low"].min()),
        "high": float(bars["high"].max()),
        "open": float(bars["open"].iloc[0]),
        "close": float(bars["close"].iloc[-1]),
        "vol_daily": vol_d,
        "vol_ann": vol_d * np.sqrt(365),
    }


def run_god_view(bars_all: pd.DataFrame, k: float) -> tuple[pd.DataFrame, pd.Series]:
    months = bars_all.groupby(pd.Grouper(freq="MS"))
    capital = INIT_CAPITAL
    rows, curves = [], []
    for mstart, bars in months:
        if len(bars) < 24 * 20:   # 略過不完整的月份
            continue
        s = month_stats(bars)
        spacing = k * s["vol_daily"]
        n = int(round(np.log(s["high"] / s["low"]) / np.log1p(spacing)))
        n = int(np.clip(n, 2, MAX_GRIDS))
        bot = GeometricGridBot(s["low"], s["high"], n, capital, fee=FEE)
        res = bot.run(bars)
        rows.append({
            "month": mstart.strftime("%Y-%m"),
            "low": s["low"], "high": s["high"],
            "range_ratio": s["high"] / s["low"],
            "vol_ann": s["vol_ann"], "n_grids": n,
            "fills": res.n_buys + res.n_sells,
            "grid_profit": res.grid_profit, "fees": res.fees_paid,
            "avg_exposure": float(res.exposure.mean()),
            "start_value": capital, "end_value": res.final_value,
            "grid_ret": res.final_value / capital - 1,
            "btc_ret": s["close"] / s["open"] - 1,
        })
        curves.append(res.value_curve)
        capital = res.final_value
    monthly = pd.DataFrame(rows)
    equity = pd.concat(curves)
    return monthly, equity


def summarize(monthly: pd.DataFrame, equity: pd.Series, k: float) -> dict:
    yrs = len(monthly) / 12
    total = monthly["end_value"].iloc[-1] / INIT_CAPITAL
    dd = (equity / equity.cummax() - 1).min()
    r = monthly["grid_ret"]
    return {
        "k": k,
        "months": len(monthly),
        "avg_n_grids": monthly["n_grids"].mean(),
        "total_return": total - 1,
        "cagr": total ** (1 / yrs) - 1,
        "monthly_mean": r.mean(), "monthly_std": r.std(),
        "sharpe_ann": r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan,
        "win_rate": (r > 0).mean(),
        "max_dd": dd,
        "total_fees": monthly["fees"].sum(),
        "avg_exposure": monthly["avg_exposure"].mean(),
    }


def main():
    bars = load_klines(SYMBOL, "1h", START, END)
    print(f"資料：{SYMBOL} 1h，{bars.index[0]} ~ {bars.index[-1]}，{len(bars)} 棒")

    summaries, all_monthly, all_equity = [], {}, {}
    for k in K_LIST:
        monthly, equity = run_god_view(bars, k)
        monthly.to_csv(RESULTS / f"god_view_monthly_k{k}.csv", index=False)
        summaries.append(summarize(monthly, equity, k))
        all_monthly[k], all_equity[k] = monthly, equity
        print(f"k={k}: 總報酬 {summaries[-1]['total_return']:+.1%}, "
              f"CAGR {summaries[-1]['cagr']:+.1%}, "
              f"勝率 {summaries[-1]['win_rate']:.0%}, "
              f"MDD {summaries[-1]['max_dd']:.1%}, "
              f"平均格數 {summaries[-1]['avg_n_grids']:.0f}")

    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS / "god_view_summary.csv", index=False)

    # ---- 淨值曲線圖 ----
    fig, ax = plt.subplots(figsize=(12, 6))
    btc_norm = bars["close"] / bars["close"].iloc[0]
    ax.plot(btc_norm.index, btc_norm, color="gray", alpha=0.6,
            label="BTC 買入持有")
    for k in K_LIST:
        eq = all_equity[k] / INIT_CAPITAL
        ax.plot(eq.index, eq, label=f"上帝視角網格 k={k}")
    ax.set_yscale("log")
    ax.set_title("上帝視角月度等比網格 vs BTC 買入持有（含 0.1% 手續費）")
    ax.set_ylabel("淨值（起始 = 1，對數刻度）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "god_view_equity.png", dpi=130)

    # ---- 診斷圖：月報酬散點與曝險 ----
    kbest = summary.loc[summary["cagr"].idxmax(), "k"]
    m = all_monthly[kbest]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(m["btc_ret"], m["grid_ret"], s=18, alpha=0.7)
    lim = max(abs(m["btc_ret"]).max(), abs(m["grid_ret"]).max()) * 1.05
    axes[0].plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="網格=BTC")
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].axvline(0, color="gray", lw=0.5)
    axes[0].set_xlabel("BTC 月報酬")
    axes[0].set_ylabel("網格月報酬")
    axes[0].set_title(f"月報酬散點（k={kbest}）")
    axes[0].legend()
    axes[1].scatter(m["vol_ann"], m["grid_ret"] - m["btc_ret"],
                    s=18, alpha=0.7, color="tab:orange")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_xlabel("該月年化日波動率")
    axes[1].set_ylabel("網格月報酬 − BTC 月報酬")
    axes[1].set_title("波動率 vs 相對 BTC 超額（k=%s）" % kbest)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "god_view_diagnostics.png", dpi=130)
    print("圖表與 CSV 已輸出至 results/")


if __name__ == "__main__":
    main()
