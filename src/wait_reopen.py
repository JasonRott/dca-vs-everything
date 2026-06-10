"""exp07：下破冷卻機制 —— 下破後不立即重開，等下一次單期投入才能重開。

動機（exp05/06 的診斷）：屯幣均價高於 DCA 的主因是崩盤段「逐腿接刀」——
每次下破立即以新價重開並半倉建倉。冷卻機制強制在跌勢中暫停部署，
現金留池吃 5% 利息，下次月投到位才重開。

比較：基準（立即重開）vs 冷卻版 vs DCA。
配置固定 ratio=50%, s=3%。全期間 + 6 時間窗 + 滾動 24 個月視窗。
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
from coin_accum import run_accum
from fetch_binance import load_klines
from grid_vs_dca import CONTRIB, WINDOWS, iter_months, run_dca, window_metrics

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp07_wait_reopen"
RESULTS.mkdir(parents=True, exist_ok=True)

RATIO, S = 0.50, 0.03
VARIANTS = {"基準(立即重開)": "immediate", "冷卻(下破等月投)": "next_contrib"}
TAGS = {"基準(立即重開)": "base", "冷卻(下破等月投)": "wait"}
COLORS = {"基準(立即重開)": "tab:gray", "冷卻(下破等月投)": "tab:blue",
          "DCA": "tab:orange"}


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")
    daily = load_klines("BTCUSDT", "1d", "2017-08-17", "2026-06-01")
    dret = np.log(daily["close"]).diff()

    nm_full = sum(1 for _ in iter_months(bars))
    dca_m, dca_curve = run_dca(bars)
    dca_qty_full = dca_m["qty_cum"].iloc[-1]
    dca_met = window_metrics(dca_curve, nm_full, CONTRIB)

    # ===== (1) 全期間 =====
    print(f"=== 全期間（ratio={RATIO:.0%}, s={S:.0%}）===")
    full_store = {}
    rows = []
    for name, mode in VARIANTS.items():
        diag, df = run_accum(bars, RATIO, S, down_reopen=mode)
        met = window_metrics(df["value"], nm_full, CONTRIB)
        full_store[name] = (diag, df)
        rows.append({"variant": name, **diag, **met,
                     "btc_vs_dca": diag["final_btc"] / dca_qty_full})
        print(f"  {name}: MOIC {met['moic']:.2f}, MDD {met['max_dd']:.0%}, "
              f"BTC {diag['final_btc']:.3f} ({diag['final_btc']/dca_qty_full:.0%} DCA), "
              f"屯幣均價 {diag['bucket_avg_cost']:,.0f}, 曝險 {diag['avg_exposure']:.0%}, "
              f"暫停時間 {diag['paused_frac']:.0%}, 利息 {diag['interest']:,.0f}")
    print(f"  DCA: MOIC {dca_met['moic']:.2f}, MDD {dca_met['max_dd']:.0%}, "
          f"BTC {dca_qty_full:.3f}, 均價 {nm_full*CONTRIB/dca_qty_full:,.0f}")
    pd.DataFrame(rows).to_csv(RESULTS / "wait_full_period.csv", index=False)

    # ===== (2) 六時間窗 =====
    print("\n=== 時間窗（MOIC | BTC比 | 屯幣均價）===")
    wrows = []
    for wname, ws, we in WINDOWS:
        sub = bars.loc[ws:we]
        nm = sum(1 for _ in iter_months(sub))
        dm, dc = run_dca(sub)
        dqty = dm["qty_cum"].iloc[-1]
        dmet = window_metrics(dc, nm, CONTRIB)
        row = {"window": wname, "dca_moic": dmet["moic"], "dca_mdd": dmet["max_dd"],
               "dca_avg_cost": nm * CONTRIB / dqty}
        line = f"  {wname}: DCA {dmet['moic']:.2f}"
        for name, mode in VARIANTS.items():
            diag, df = run_accum(sub, RATIO, S, down_reopen=mode)
            met = window_metrics(df["value"], nm, CONTRIB)
            tag = TAGS[name]
            row[f"{tag}_moic"] = met["moic"]
            row[f"{tag}_mdd"] = met["max_dd"]
            row[f"{tag}_btc_ratio"] = diag["final_btc"] / dqty
            row[f"{tag}_avg_cost"] = diag["bucket_avg_cost"]
            line += (f" | {name} {met['moic']:.2f}, "
                     f"{diag['final_btc']/dqty:.0%}, {diag['bucket_avg_cost']:,.0f}")
        wrows.append(row)
        print(line)
    pd.DataFrame(wrows).to_csv(RESULTS / "wait_windows.csv", index=False)

    # ===== (3) 滾動 24 個月 =====
    starts = pd.date_range("2018-01-01", "2024-06-01", freq="3MS", tz="UTC")
    roll = []
    for st in starts:
        en = st + pd.DateOffset(months=24)
        sub = bars.loc[st:en]
        nm = sum(1 for _ in iter_months(sub))
        if nm < 24:
            continue
        dm, dc = run_dca(sub)
        dqty = dm["qty_cum"].iloc[-1]
        dmet = window_metrics(dc, nm, CONTRIB)
        vol = float(dret[(dret.index >= st) & (dret.index < en)].std() * np.sqrt(365))
        r = {"start": st, "win_vol": vol,
             "dca_moic": dmet["moic"], "dca_mdd": dmet["max_dd"]}
        for name, mode in VARIANTS.items():
            diag, df = run_accum(sub, RATIO, S, down_reopen=mode)
            met = window_metrics(df["value"], nm, CONTRIB)
            tag = TAGS[name]
            r[f"{tag}_moic"] = met["moic"]
            r[f"{tag}_mdd"] = met["max_dd"]
            r[f"{tag}_btc_ratio"] = diag["final_btc"] / dqty
            r[f"{tag}_avg_cost_ratio"] = diag["bucket_avg_cost"] / (nm * CONTRIB / dqty)
            r[f"{tag}_expo"] = diag["avg_exposure"]
        roll.append(r)
    rdf = pd.DataFrame(roll)
    rdf.to_csv(RESULTS / "wait_rolling24m.csv", index=False)

    yr = rdf.assign(yr=rdf["start"].dt.year).groupby("yr")[
        ["win_vol", "dca_moic", "base_moic", "wait_moic",
         "base_btc_ratio", "wait_btc_ratio",
         "base_avg_cost_ratio", "wait_avg_cost_ratio"]].mean().round(3)
    print("\n=== 滾動視窗依起始年平均 ===")
    print(yr.to_string())
    yr.to_csv(RESULTS / "wait_rolling_by_year.csv")

    # ===== 圖表 =====
    # 圖1：滾動視窗 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.plot(rdf["start"], rdf["dca_moic"], "-o", ms=3, color=COLORS["DCA"], label="DCA")
    for name, tag in TAGS.items():
        ax.plot(rdf["start"], rdf[f"{tag}_moic"], "-o", ms=3,
                color=COLORS[name], label=name)
    ax.axhline(1, color="gray", lw=0.8)
    ax.set_title("滾動 24 個月 MOIC"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for name, tag in TAGS.items():
        ax.plot(rdf["start"], rdf[f"{tag}_btc_ratio"], "-o", ms=3,
                color=COLORS[name], label=name)
    ax.axhline(1, color=COLORS["DCA"], lw=1.2, label="DCA = 1")
    ax.set_title("期末持幣量 / DCA"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for name, tag in TAGS.items():
        ax.plot(rdf["start"], rdf[f"{tag}_avg_cost_ratio"], "-o", ms=3,
                color=COLORS[name], label=name)
    ax.axhline(1, color=COLORS["DCA"], lw=1.2, label="DCA 均價 = 1")
    ax.set_title("屯幣均價 / DCA 均價（<1 = 比 DCA 便宜）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(rdf["start"], rdf["dca_mdd"], "-o", ms=3, color=COLORS["DCA"], label="DCA")
    for name, tag in TAGS.items():
        ax.plot(rdf["start"], rdf[f"{tag}_mdd"], "-o", ms=3,
                color=COLORS[name], label=name)
    ax.set_title("最大回撤"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for axx in axes.flat:
        axx.set_xlabel("視窗起始月")
    fig.suptitle(f"exp07：下破冷卻 vs 立即重開（滾動 24 個月，ratio={RATIO:.0%}, s={S:.0%}）",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(RESULTS / "wait_rolling.png", dpi=130)

    # 圖2：曝險走勢對比
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(bars.index, bars["close"], color="gray", lw=0.8)
    axes[0].set_yscale("log"); axes[0].set_ylabel("BTC 價格"); axes[0].grid(alpha=0.3)
    axes[0].set_title("曝險走勢：立即重開 vs 下破冷卻")
    for axx, name in zip(axes[1:], VARIANTS):
        _, df = full_store[name]
        axx.fill_between(df.index, df["exposure"], color=COLORS[name], alpha=0.55)
        axx.axhline(1.0, color="gray", lw=0.5)
        axx.set_ylim(0, 1.05); axx.set_ylabel(name, fontsize=9); axx.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "wait_exposure.png", dpi=130)

    # 圖3：法幣淨值
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in VARIANTS:
        _, df = full_store[name]
        ax.plot(df.index, df["value"], color=COLORS[name], lw=1.1, label=name)
    ax.plot(dca_curve.index, dca_curve.values, color=COLORS["DCA"], lw=1.2,
            alpha=0.85, label="DCA")
    cidx = pd.date_range(dca_curve.index[0], periods=nm_full, freq="MS")
    ax.plot(cidx, np.arange(1, nm_full + 1) * CONTRIB, "k--", lw=1, label="累計投入")
    ax.set_yscale("log"); ax.set_ylabel("淨值 (USDT)")
    ax.set_title("法幣淨值：下破冷卻 vs 立即重開 vs DCA（對數刻度）")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "wait_value.png", dpi=130)

    print("\n輸出：results/exp07_wait_reopen/")


if __name__ == "__main__":
    main()
