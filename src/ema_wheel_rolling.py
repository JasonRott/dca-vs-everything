"""exp14 滾動視窗穩健性：EMA 雙輪 vs DCA / exp10 / exp13。

24 個月視窗、起點 2021-06 ~ 2024-06（步進 3 個月，受 DVOL 起始限制）。
重點：exp14 的「報酬換回撤」交換在各 regime 的勝率與幅度。
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
from dca_covered_call import run_dca_cc
from dual_wheel import run_dual_wheel
from ema_accum import build_ema
from ema_wheel import run_ema_wheel
from fetch_binance import load_klines
from fetch_deribit import load_dvol
from grid_vs_dca import CONTRIB, iter_months, window_metrics

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False
RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp14_ema_wheel"


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")
    daily = load_klines("BTCUSDT", "1d", "2017-08-17", "2026-06-01")
    ema = build_ema(daily, bars.index)
    iv = load_dvol()["close"].shift(1)

    starts = pd.date_range("2021-06-01", "2024-06-01", freq="3MS", tz="UTC")
    roll = []
    for st in starts:
        sub = bars.loc[st:st + pd.DateOffset(months=24)]
        nm = sum(1 for _ in iter_months(sub))
        if nm < 24:
            continue
        r = {"start": st}
        _, ddf = run_dca_cc(sub, 0.0)
        met = window_metrics(ddf["value"], nm, CONTRIB)
        dqty = float(ddf["btc"].iloc[-1])
        r["dca_moic"], r["dca_irr"], r["dca_mdd"] = \
            met["moic"], met["irr_ann"], met["max_dd"]
        d10, df10 = run_dca_cc(sub, 0.25, 0.03, "above_ema", ema, iv)
        met = window_metrics(df10["value"], nm, CONTRIB)
        r["exp10_moic"], r["exp10_irr"], r["exp10_mdd"] = \
            met["moic"], met["irr_ann"], met["max_dd"]
        r["exp10_btc"] = d10["final_btc"] / dqty
        d13, df13 = run_dual_wheel(sub, (0.50, 0.70), iv, buy_target="dca")
        met = window_metrics(df13["value"], nm, CONTRIB)
        r["exp13_moic"], r["exp13_irr"], r["exp13_mdd"] = \
            met["moic"], met["irr_ann"], met["max_dd"]
        for frac, tag in [(0.10, "w10"), (0.25, "w25")]:
            diag, df = run_ema_wheel(sub, ema, iv, frac)
            met = window_metrics(df["value"], nm, CONTRIB)
            r[f"{tag}_moic"], r[f"{tag}_irr"], r[f"{tag}_mdd"] = \
                met["moic"], met["irr_ann"], met["max_dd"]
            r[f"{tag}_btc"] = diag["final_btc"] / dqty
        roll.append(r)
    rdf = pd.DataFrame(roll)
    rdf.to_csv(RESULTS / "rolling24m.csv", index=False)

    show = rdf.assign(start=rdf["start"].dt.strftime("%Y-%m"))
    print("=== 滾動 24 個月：MOIC ===")
    print(show[["start", "dca_moic", "exp10_moic", "exp13_moic",
                "w10_moic", "w25_moic"]].round(3).to_string(index=False))
    print("\n=== 滾動 24 個月：MDD ===")
    print(show[["start", "dca_mdd", "exp10_mdd", "exp13_mdd",
                "w10_mdd", "w25_mdd"]].round(3).to_string(index=False))
    print("\n=== 勝率摘要（exp14 frac=10% vs 各對照）===")
    for tag, name in [("dca", "DCA"), ("exp10", "exp10"), ("exp13", "exp13")]:
        wr_m = (rdf["w10_moic"] > rdf[f"{tag}_moic"]).mean()
        wr_d = (rdf["w10_mdd"] > rdf[f"{tag}_mdd"]).mean()
        cal_w10 = rdf["w10_irr"] / rdf["w10_mdd"].abs()
        cal_x = rdf[f"{tag}_irr"] / rdf[f"{tag}_mdd"].abs()
        wr_c = (cal_w10 > cal_x).mean()
        print(f"  vs {name}: MOIC 勝率 {wr_m:.0%} | MDD 較淺比例 {wr_d:.0%} | "
              f"Calmar 勝率 {wr_c:.0%}")
    print(f"  exp14 w10 幣量/DCA：平均 {rdf['w10_btc'].mean():.0%}"
          f"（範圍 {rdf['w10_btc'].min():.0%}~{rdf['w10_btc'].max():.0%}）")
    print(f"  exp10 幣量/DCA：平均 {rdf['exp10_btc'].mean():.0%}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    COLS = [("dca", "tab:orange", "DCA"), ("exp10", "tab:red", "exp10 CC25EMA"),
            ("exp13", "tab:gray", "exp13 帶控"), ("w10", "tab:blue", "exp14 w10%"),
            ("w25", "tab:green", "exp14 w25%")]
    for tag, color, lbl in COLS:
        axes[0, 0].plot(rdf["start"], rdf[f"{tag}_moic"], "-o", ms=3,
                        color=color, label=lbl)
        axes[0, 1].plot(rdf["start"], rdf[f"{tag}_mdd"], "-o", ms=3,
                        color=color, label=lbl)
        axes[1, 0].plot(rdf["start"], rdf[f"{tag}_irr"] / rdf[f"{tag}_mdd"].abs(),
                        "-o", ms=3, color=color, label=lbl)
    axes[0, 0].axhline(1, color="gray", lw=0.8)
    axes[0, 0].set_title("MOIC")
    axes[0, 1].set_title("最大回撤")
    axes[1, 0].set_title("Calmar (IRR/|MDD|)")
    for tag, color, lbl in COLS[1:]:
        if f"{tag}_btc" in rdf:
            axes[1, 1].plot(rdf["start"], rdf[f"{tag}_btc"], "-o", ms=3,
                            color=color, label=lbl)
    axes[1, 1].axhline(1, color="tab:orange", lw=1.2, label="DCA=1")
    axes[1, 1].set_title("期末持幣量 / DCA")
    for ax in axes.flat:
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlabel("視窗起始月")
    fig.suptitle("exp14 滾動視窗：EMA 雙輪 vs 全部對照", fontsize=13)
    fig.tight_layout(); fig.savefig(RESULTS / "rolling.png", dpi=130)
    print("\n輸出：results/exp14_ema_wheel/rolling.png, rolling24m.csv")


if __name__ == "__main__":
    main()
