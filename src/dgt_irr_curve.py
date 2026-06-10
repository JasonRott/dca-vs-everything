"""exp12b：重算 DGT 的 IRR「時間序列」——驗證論文 60-70% 是否為曲線峰值。

論文 Fig.3/5 的 IRR 是隨時間演進的年化曲線。依其記帳法在時點 t：
    IRR(t) = (V(t) / money_input(t)) ** (12 / 已經過月數) − 1
其中 V(t) = USDT + COIN×price（兩版本：含/不含進行中網格的帳面本金），
money_input(t) = 截至 t 的累計注資。早期（經過月數小）的年化會被指數放大。

若峰值確實達 60-70% → 修正本研究先前「頭條數字不可重現」的判斷為
「可重現為時間序列峰值；但作為頭條數字依賴注資當期初的記帳與年化放大」。
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
from dgt_faithful import PRINCIPAL, run_dgt_paper
from fetch_binance import load_klines

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False
RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp12_dgt_faithful"

SIZES = [0.005, 0.01, 0.02, 0.05, 0.10]
MS = [2, 5, 10]
BURN_DAYS = 60          # 避開最初年化爆炸的觀察起點（敏感度另列 30/90）


def irr_curve(r, t0):
    curve = r["curve"]                       # 每小時 V(t)，含網格帳面本金
    flows = r["flows"]
    ft = pd.DatetimeIndex([f[0] for f in flows])
    fa = np.cumsum([f[1] for f in flows])
    mi_raw = pd.Series(fa, index=ft).groupby(level=0).last()  # 同棒多筆注資合併
    mi = mi_raw.reindex(curve.index, method="ffill")
    months = (curve.index - t0).total_seconds() / 86400 / 30.44
    months = np.maximum(months.values, 1e-6)
    moic = curve.values / mi.values
    with np.errstate(over="ignore"):
        irr_incl = moic ** (12 / months) - 1
        moic_ex = np.maximum(curve.values - PRINCIPAL, 1e-9) / mi.values
        irr_excl = moic_ex ** (12 / months) - 1
    return pd.DataFrame({"irr_incl": irr_incl, "irr_excl": irr_excl,
                         "moic": moic}, index=curve.index)


def main():
    bars = load_klines("BTCUSDT", "1m", "2021-01-01", "2024-08-01")
    t0 = bars.index[0]
    rows, curves = [], {}
    for size in SIZES:
        for m in MS:
            r = run_dgt_paper(bars, size, m)
            df = irr_curve(r, t0)
            curves[(size, m)] = df
            row = {"size": size, "m": m,
                   "final_irr": df["irr_incl"].iloc[-1]}
            for burn in (30, 60, 90):
                sub = df[df.index >= t0 + pd.Timedelta(days=burn)]
                row[f"peak_irr_burn{burn}"] = float(sub["irr_incl"].max())
                row[f"peak_t_burn{burn}"] = sub["irr_incl"].idxmax() \
                    .strftime("%Y-%m")
            rows.append(row)
            print(f"size={size:.3f}, m={m}: 期末IRR {row['final_irr']:+.1%} | "
                  f"峰值IRR(burn30/60/90天) "
                  f"{row['peak_irr_burn30']:+.0%}@{row['peak_t_burn30']} / "
                  f"{row['peak_irr_burn60']:+.0%}@{row['peak_t_burn60']} / "
                  f"{row['peak_irr_burn90']:+.0%}@{row['peak_t_burn90']}")
    sdf = pd.DataFrame(rows)
    sdf.to_csv(RESULTS / "dgt_irr_curve_peaks.csv", index=False)

    n6070 = ((sdf["peak_irr_burn60"] >= 0.5) & (sdf["peak_irr_burn60"] <= 1.0)).sum()
    print(f"\n峰值落在 +50%~+100% 區間的配置數（burn 60 天）：{n6070}/15")
    print(f"峰值中位數（burn 60 天）：{sdf['peak_irr_burn60'].median():+.0%}")

    # 圖：代表配置的 IRR(t) 曲線
    fig, ax = plt.subplots(figsize=(13, 6))
    for (size, m), color in [((0.005, 2), "tab:blue"), ((0.02, 5), "tab:green"),
                             ((0.05, 10), "tab:purple"), ((0.10, 5), "tab:red")]:
        df = curves[(size, m)]
        sub = df[df.index >= t0 + pd.Timedelta(days=BURN_DAYS)]
        ax.plot(sub.index, sub["irr_incl"] * 100, color=color, lw=1.0,
                label=f"size={size:.1%}, m={m}")
    ax.axhspan(60, 70, color="orange", alpha=0.2, label="論文宣稱 60-70%")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("IRR(t) %（論文式記帳，年化自起始日）")
    ax.set_title("DGT 官方引擎：IRR 時間序列（觀察起點 = 第 60 天）")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "dgt_irr_curve.png", dpi=130)
    print("\n輸出：dgt_irr_curve.png, dgt_irr_curve_peaks.csv")


if __name__ == "__main__":
    main()
