"""exp08：屯幣版第一部分（不含選擇權）總結。

新增變體：EMA 條件式冷卻——下破時價格 < 200EMA 才冷卻至下次月投，
價格 ≥ EMA 則立即重開（exp06/07 的數據驅動組合）。

主分析視窗改為 2023-07-01 起（BTC ~30k，位於 2022-11 週期底與 2024-03 頂
之間的半山腰，非頂非底、低波動 regime、35 個完整月）。
另保留滾動 24 個月視窗供 regime 穩健性檢驗。

四策略：DCA / 基準(立即重開) / 全程冷卻 / EMA 條件式冷卻。
配置固定 ratio=50%, s=3%。
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
from ema_accum import build_ema
from fetch_binance import load_klines
from grid_vs_dca import CONTRIB, iter_months, run_dca, window_metrics

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp08_final_summary"
RESULTS.mkdir(parents=True, exist_ok=True)

RATIO, S = 0.50, 0.03
MAIN_START = "2023-07-01"
VARIANTS = {
    "基準(立即重開)": dict(down_reopen="immediate"),
    "全程冷卻": dict(down_reopen="next_contrib"),
    "EMA條件式冷卻": dict(down_reopen="ema_conditional"),
}
TAGS = {"基準(立即重開)": "base", "全程冷卻": "wait", "EMA條件式冷卻": "hybrid"}
COLORS = {"基準(立即重開)": "tab:gray", "全程冷卻": "tab:blue",
          "EMA條件式冷卻": "tab:green", "DCA": "tab:orange"}


def eval_window(sub: pd.DataFrame, ema: pd.Series) -> tuple[dict, dict]:
    """回傳 ({tag: 指標}, {name: df})，含 DCA。"""
    nm = sum(1 for _ in iter_months(sub))
    dm, dc = run_dca(sub)
    dqty = float(dm["qty_cum"].iloc[-1])
    dmet = window_metrics(dc, nm, CONTRIB)
    out = {"dca": {"moic": dmet["moic"], "mdd": dmet["max_dd"],
                   "btc": dqty, "avg_cost": nm * CONTRIB / dqty,
                   "expo": 1.0}}
    dfs = {"DCA": dc}
    for name, kw in VARIANTS.items():
        diag, df = run_accum(sub, RATIO, S, ema=ema, **kw)
        met = window_metrics(df["value"], nm, CONTRIB)
        out[TAGS[name]] = {
            "moic": met["moic"], "mdd": met["max_dd"],
            "btc": diag["final_btc"], "btc_ratio": diag["final_btc"] / dqty,
            "avg_cost": diag["bucket_avg_cost"],
            "cost_ratio": diag["bucket_avg_cost"] / (nm * CONTRIB / dqty),
            "expo": diag["avg_exposure"], "paused": diag["paused_frac"],
        }
        dfs[name] = df
    return out, dfs


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")
    daily = load_klines("BTCUSDT", "1d", "2017-08-17", "2026-06-01")
    ema = build_ema(daily, bars.index)
    dret = np.log(daily["close"]).diff()

    # ===== (1) 主視窗：2023-07 起 =====
    main_bars = bars.loc[MAIN_START:]
    res, dfs = eval_window(main_bars, ema)
    nm = sum(1 for _ in iter_months(main_bars))
    print(f"=== 主視窗 {MAIN_START} ~ 2026-05（{nm} 個月，月投 1000）===")
    mrows = []
    for tag in ["dca", "base", "wait", "hybrid"]:
        r = res[tag]
        mrows.append({"strategy": tag, **r})
        extra = ("" if tag == "dca" else
                 f", BTC比 {r['btc_ratio']:.0%}, 均價比 {r['cost_ratio']:.2f}, "
                 f"暫停 {r['paused']:.0%}")
        print(f"  {tag}: MOIC {r['moic']:.3f}, MDD {r['mdd']:.0%}, "
              f"BTC {r['btc']:.4f}, 均價 {r['avg_cost']:,.0f}, "
              f"曝險 {r['expo']:.0%}{extra}")
    pd.DataFrame(mrows).to_csv(RESULTS / "main_window.csv", index=False)

    # ===== (2) 滾動 24 個月視窗 =====
    starts = pd.date_range("2018-01-01", "2024-06-01", freq="3MS", tz="UTC")
    roll = []
    for st in starts:
        en = st + pd.DateOffset(months=24)
        sub = bars.loc[st:en]
        if sum(1 for _ in iter_months(sub)) < 24:
            continue
        r_, _ = eval_window(sub, ema)
        vol = float(dret[(dret.index >= st) & (dret.index < en)].std() * np.sqrt(365))
        row = {"start": st, "win_vol": vol, "dca_moic": r_["dca"]["moic"],
               "dca_mdd": r_["dca"]["mdd"]}
        for tag in ["base", "wait", "hybrid"]:
            row[f"{tag}_moic"] = r_[tag]["moic"]
            row[f"{tag}_mdd"] = r_[tag]["mdd"]
            row[f"{tag}_btc_ratio"] = r_[tag]["btc_ratio"]
            row[f"{tag}_cost_ratio"] = r_[tag]["cost_ratio"]
        roll.append(row)
    rdf = pd.DataFrame(roll)
    rdf.to_csv(RESULTS / "rolling24m.csv", index=False)
    yr = rdf.assign(yr=rdf["start"].dt.year).groupby("yr")[
        ["win_vol", "dca_moic", "base_moic", "wait_moic", "hybrid_moic",
         "base_btc_ratio", "wait_btc_ratio", "hybrid_btc_ratio",
         "base_cost_ratio", "wait_cost_ratio", "hybrid_cost_ratio"]].mean().round(3)
    print("\n=== 滾動視窗依起始年平均 ===")
    print(yr.to_string())
    yr.to_csv(RESULTS / "rolling_by_year.csv")

    # ===== 圖表 =====
    # 圖1：主視窗總覽（淨值 / 持幣量 / 曝險）
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1, 1])

    ax = fig.add_subplot(gs[0, :])
    for name, df in dfs.items():
        if name == "DCA":
            ax.plot(df.index, df.values, color=COLORS[name], lw=1.4, label=name)
        else:
            ax.plot(df.index, df["value"], color=COLORS[name], lw=1.2, label=name)
    cidx = pd.date_range(dfs["DCA"].index[0], periods=nm, freq="MS")
    ax.plot(cidx, np.arange(1, nm + 1) * CONTRIB, "k--", lw=1, label="累計投入")
    ax.set_title(f"主視窗 {MAIN_START}~ ：法幣淨值（月投 1000）")
    ax.set_ylabel("USDT"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    dq = dfs["DCA"] / main_bars["close"].reindex(dfs["DCA"].index)
    ax.plot(dq.index, dq.values, color=COLORS["DCA"], lw=1.3, label="DCA")
    for name in VARIANTS:
        df = dfs[name]
        ax.plot(df.index, df["btc"], color=COLORS[name], lw=1.1, label=name)
    ax.set_title("累積持幣量 (BTC)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(main_bars.index, main_bars["close"], color="gray", lw=0.8, label="BTC")
    e_sub = ema.loc[main_bars.index[0]:]
    ax.plot(e_sub.index, e_sub.values, color="tab:purple", lw=0.9, label="200日EMA")
    ax.set_title("價格與 200 日 EMA"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax1 = fig.add_subplot(gs[2, :])
    for name in VARIANTS:
        df = dfs[name]
        ax1.plot(df.index, df["exposure"], color=COLORS[name], lw=1.0,
                 alpha=0.9, label=name)
    ax1.axhline(1.0, color=COLORS["DCA"], lw=1.2, label="DCA = 1")
    ax1.set_ylim(0, 1.05); ax1.set_title("曝險走勢（BTC 市值 / 總淨值）")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "main_window_overview.png", dpi=130)

    # 圖2：滾動視窗總結（MOIC / 幣量比 / 均價比 / MDD）
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
        ax.plot(rdf["start"], rdf[f"{tag}_cost_ratio"], "-o", ms=3,
                color=COLORS[name], label=name)
    ax.axhline(1, color=COLORS["DCA"], lw=1.2, label="DCA 均價 = 1")
    ax.set_title("屯幣均價 / DCA 均價（<1 = 更便宜）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(rdf["start"], rdf["dca_mdd"], "-o", ms=3, color=COLORS["DCA"], label="DCA")
    for name, tag in TAGS.items():
        ax.plot(rdf["start"], rdf[f"{tag}_mdd"], "-o", ms=3,
                color=COLORS[name], label=name)
    ax.set_title("最大回撤"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for axx in axes.flat:
        axx.set_xlabel("視窗起始月")
    fig.suptitle("exp08：屯幣版總結 — 滾動 24 個月視窗（ratio=50%, s=3%）", fontsize=13)
    fig.tight_layout(); fig.savefig(RESULTS / "rolling_summary.png", dpi=130)

    print("\n輸出：results/exp08_final_summary/")


if __name__ == "__main__":
    main()
