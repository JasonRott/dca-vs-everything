"""統一比較報告：主視窗（2023-07 起）所有策略家族 + 基線的一致性重算。

策略集：純現金5%、DCA、屯幣網格家族（基準/全程冷卻/EMA條件冷卻）、
covered call 家族（常態輪動 2%/5%、EMA上才賣 3%）。
指標：MOIC、年化IRR、MDD、Calmar(IRR/|MDD|)、期末BTC、屯幣均價、平均曝險。
輸出：reports/figures/ 下的總表 CSV 與風險-報酬散點圖。
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
from coin_accum import HOURLY_R, run_accum
from dca_covered_call import run_dca_cc
from ema_accum import build_ema
from fetch_binance import load_klines
from fetch_deribit import load_dvol
from grid_vs_dca import CONTRIB, iter_months, run_dca, window_metrics

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent.parent / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

RATIO, S = 0.50, 0.03
MAIN_START = "2023-07-01"


def cash_baseline(bars: pd.DataFrame) -> tuple[float, int]:
    """月投 1000、全程 5% 逐時複利的期末值。"""
    pool = 0.0
    nm = 0
    for _, mbars in iter_months(bars):
        pool += CONTRIB
        nm += 1
        pool *= (1 + HOURLY_R) ** len(mbars)
    return pool, nm


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")
    daily = load_klines("BTCUSDT", "1d", "2017-08-17", "2026-06-01")
    ema = build_ema(daily, bars.index)
    iv = load_dvol()["close"].shift(1)
    sub = bars.loc[MAIN_START:]
    nm = sum(1 for _ in iter_months(sub))

    rows = []

    # 純現金 5%
    final_cash, _ = cash_baseline(sub)
    met = window_metrics(pd.Series([final_cash]), nm, CONTRIB)  # 只取 moic/irr 用
    rows.append({"strategy": "純現金 5% APY", "family": "基線",
                 "moic": final_cash / (nm * CONTRIB),
                 "irr_ann": window_metrics(
                     pd.Series([final_cash], index=[sub.index[-1]]),
                     nm, CONTRIB)["irr_ann"],
                 "max_dd": 0.0, "btc": 0.0, "avg_cost": np.nan,
                 "avg_expo": 0.0})

    # DCA
    dm, dc = run_dca(sub)
    dqty = float(dm["qty_cum"].iloc[-1])
    met = window_metrics(dc, nm, CONTRIB)
    rows.append({"strategy": "DCA", "family": "基線",
                 "moic": met["moic"], "irr_ann": met["irr_ann"],
                 "max_dd": met["max_dd"], "btc": dqty,
                 "avg_cost": nm * CONTRIB / dqty, "avg_expo": 1.0})

    configs = [
        ("屯幣網格-基準", dict(), "屯幣網格"),
        ("屯幣網格-全程冷卻", dict(down_reopen="next_contrib"), "屯幣網格"),
        ("屯幣網格-EMA條件冷卻", dict(down_reopen="ema_conditional"), "屯幣網格"),
        ("call常態輪動 otm2%", dict(call_frac=0.25, call_otm=0.02,
                                   call_trigger="always", iv=iv), "covered call"),
        ("call常態輪動 otm5%", dict(call_frac=0.25, call_otm=0.05,
                                   call_trigger="always", iv=iv), "covered call"),
        ("call EMA上才賣 otm3%", dict(call_frac=0.25, call_otm=0.03,
                                     call_trigger="above_ema", iv=iv), "covered call"),
    ]
    for name, kw, fam in configs:
        diag, df = run_accum(sub, RATIO, S, ema=ema, **kw)
        met = window_metrics(df["value"], nm, CONTRIB)
        rows.append({"strategy": name, "family": fam,
                     "moic": met["moic"], "irr_ann": met["irr_ann"],
                     "max_dd": met["max_dd"], "btc": diag["final_btc"],
                     "avg_cost": diag["bucket_avg_cost"],
                     "avg_expo": diag["avg_exposure"],
                     "premium": diag.get("premium_total", 0.0)})

    # DCA + covered call（exp10：權利金與履約款月度再投入）
    for name, f, trig in [("DCA+CC 10% 常態", 0.10, "always"),
                          ("DCA+CC 25% 常態", 0.25, "always"),
                          ("DCA+CC 25% EMA上", 0.25, "above_ema")]:
        diag, df = run_dca_cc(sub, f, 0.03, trig, ema, iv)
        met = window_metrics(df["value"], nm, CONTRIB)
        rows.append({"strategy": name, "family": "DCA+CC",
                     "moic": met["moic"], "irr_ann": met["irr_ann"],
                     "max_dd": met["max_dd"], "btc": diag["final_btc"],
                     "avg_cost": np.nan, "avg_expo": 1.0,
                     "premium": diag["premium_total"]})

    df = pd.DataFrame(rows)
    df["btc_ratio"] = df["btc"] / dqty
    df["calmar"] = df["irr_ann"] / df["max_dd"].abs().replace(0, np.nan)
    df["excess_irr_vs_cash"] = df["irr_ann"] - df.loc[0, "irr_ann"]
    df.to_csv(OUT / "comparison_table.csv", index=False)
    pd.set_option("display.width", 200)
    print(df[["strategy", "moic", "irr_ann", "max_dd", "calmar",
              "btc_ratio", "avg_expo"]].round(3).to_string(index=False))

    # 風險-報酬散點
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = {"基線": "tab:orange", "屯幣網格": "tab:gray",
            "covered call": "tab:green", "DCA+CC": "tab:red"}
    for _, r in df.iterrows():
        x = abs(r["max_dd"]) * 100
        y = r["irr_ann"] * 100
        ax.scatter(x, y, s=110, color=cmap[r["family"]], zorder=3,
                   edgecolor="k", lw=0.6)
        size = r["btc_ratio"]
        ax.annotate(f"{r['strategy']}\n(幣量 {size:.0%})" if size > 0
                    else r["strategy"],
                    (x, y), textcoords="offset points", xytext=(8, 6),
                    fontsize=8.5)
    ax.axhline(df.loc[0, "irr_ann"] * 100, color="tab:orange", lw=0.8,
               ls="--", label="無風險 5% 水準")
    ax.set_xlabel("最大回撤 |MDD| (%)")
    ax.set_ylabel("年化 IRR (%)")
    ax.set_title(f"風險-報酬全景（主視窗 {MAIN_START}~，月投 1000，{nm} 個月）\n"
                 "點標注 = 期末持幣量相對 DCA")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "risk_return_scatter.png", dpi=130)
    print(f"\n輸出：{OUT}")


if __name__ == "__main__":
    main()
