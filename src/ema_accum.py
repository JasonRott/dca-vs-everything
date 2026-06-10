"""exp06：屯幣版網格的滾動視窗檢驗 + 200 日 EMA 建倉濾鏡變體。

動機：
- exp05 全期間結果受 BTC 早期高波動主導，需用滾動 24 個月視窗看「波動率
  收斂後」策略還剩多少優勢。
- EMA 變體（網格照常開，僅初始半倉建倉受濾鏡控制）：
  A "no_open_below"：價格 < 200EMA 時不建底倉（跌勢不於腿頂接刀）
  B "no_open_above"：價格 > 200EMA 時不建底倉（漲勢不追高建倉）

EMA 無前視：以日線收盤計算 ewm(span=200)，shift(1) 後對齊小時索引。
配置固定 ratio=50%, s=3%（exp05 已證明參數不敏感）。
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

RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp06_ema_rolling"
RESULTS.mkdir(parents=True, exist_ok=True)

RATIO, S = 0.50, 0.03
VARIANTS = {"基準(永遠半倉)": None,
            "A: EMA下不建倉": "no_open_below",
            "B: EMA上不建倉": "no_open_above"}
COLORS = {"基準(永遠半倉)": "tab:gray", "A: EMA下不建倉": "tab:blue",
          "B: EMA上不建倉": "tab:red", "DCA": "tab:orange"}


def build_ema(daily: pd.DataFrame, hourly_index: pd.DatetimeIndex) -> pd.Series:
    ema_d = daily["close"].ewm(span=200, adjust=False).mean().shift(1)
    return ema_d.reindex(hourly_index, method="ffill")


def metrics_pack(bars_sub, variant_mode, ema, dca_qty, nm):
    diag, df = run_accum(bars_sub, RATIO, S, ema=ema, ema_mode=variant_mode)
    met = window_metrics(df["value"], nm, CONTRIB)
    return diag, df, met


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")
    daily = load_klines("BTCUSDT", "1d", "2017-08-17", "2026-06-01")
    ema = build_ema(daily, bars.index)
    dret = np.log(daily["close"]).diff()

    nm_full = sum(1 for _ in iter_months(bars))
    dca_m, dca_curve = run_dca(bars)
    dca_qty_full = dca_m["qty_cum"].iloc[-1]

    # ===== (1) 全期間：三變體雙重記帳 =====
    print(f"=== 全期間（ratio={RATIO:.0%}, s={S:.0%}）===")
    full_store = {}
    rows = []
    for name, mode in VARIANTS.items():
        diag, df, met = metrics_pack(bars, mode, ema, dca_qty_full, nm_full)
        full_store[name] = (diag, df)
        rows.append({"variant": name, **diag, **met,
                     "btc_vs_dca": diag["final_btc"] / dca_qty_full})
        print(f"  {name}: MOIC {met['moic']:.2f}, MDD {met['max_dd']:.0%}, "
              f"BTC {diag['final_btc']:.3f} ({diag['final_btc']/dca_qty_full:.0%} DCA), "
              f"屯幣均價 {diag['bucket_avg_cost']:,.0f}, 曝險 {diag['avg_exposure']:.0%}, "
              f"建倉開格 {diag['opens_with_pos']}/免建倉 {diag['opens_without_pos']}")
    dca_met_full = window_metrics(dca_curve, nm_full, CONTRIB)
    print(f"  DCA: MOIC {dca_met_full['moic']:.2f}, MDD {dca_met_full['max_dd']:.0%}, "
          f"BTC {dca_qty_full:.3f}, 均價 {nm_full*CONTRIB/dca_qty_full:,.0f}")
    pd.DataFrame(rows).to_csv(RESULTS / "ema_full_period.csv", index=False)

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
            diag, df, met = metrics_pack(sub, mode, ema, dqty, nm)
            tag = {"基準(永遠半倉)": "base", "A: EMA下不建倉": "A",
                   "B: EMA上不建倉": "B"}[name]
            row[f"{tag}_moic"] = met["moic"]
            row[f"{tag}_mdd"] = met["max_dd"]
            row[f"{tag}_btc_ratio"] = diag["final_btc"] / dqty
            row[f"{tag}_avg_cost"] = diag["bucket_avg_cost"]
            line += f" | {name} {met['moic']:.2f}, {diag['final_btc']/dqty:.0%}"
        wrows.append(row)
        print(line)
    pd.DataFrame(wrows).to_csv(RESULTS / "ema_windows.csv", index=False)

    # ===== (3) 滾動 24 個月視窗 =====
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
            diag, df, met = metrics_pack(sub, mode, ema, dqty, nm)
            tag = {"基準(永遠半倉)": "base", "A: EMA下不建倉": "A",
                   "B: EMA上不建倉": "B"}[name]
            r[f"{tag}_moic"] = met["moic"]
            r[f"{tag}_mdd"] = met["max_dd"]
            r[f"{tag}_btc_ratio"] = diag["final_btc"] / dqty
            r[f"{tag}_expo"] = diag["avg_exposure"]
        roll.append(r)
    rdf = pd.DataFrame(roll)
    rdf.to_csv(RESULTS / "ema_rolling24m.csv", index=False)

    # ===== 圖表 =====
    TAGS = {"基準(永遠半倉)": "base", "A: EMA下不建倉": "A", "B: EMA上不建倉": "B"}

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
    ax.set_title("滾動 24 個月：期末持幣量 / DCA 持幣量")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(rdf["start"], rdf["dca_mdd"], "-o", ms=3, color=COLORS["DCA"], label="DCA")
    for name, tag in TAGS.items():
        ax.plot(rdf["start"], rdf[f"{tag}_mdd"], "-o", ms=3,
                color=COLORS[name], label=name)
    ax.set_title("滾動 24 個月 最大回撤"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.bar(rdf["start"], rdf["win_vol"], width=80, color="tab:cyan", alpha=0.7)
    ax.set_title("視窗內年化日波動率（波動 regime 對照）")
    ax.grid(alpha=0.3)
    for axx in axes.flat:
        axx.set_xlabel("視窗起始月")
    fig.suptitle(f"exp06：滾動視窗檢驗（ratio={RATIO:.0%}, s={S:.0%}）", fontsize=13)
    fig.tight_layout(); fig.savefig(RESULTS / "ema_rolling.png", dpi=130)

    # 圖2：曝險走勢對比（全期間）
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(bars.index, bars["close"], color="gray", lw=0.8, label="BTC")
    axes[0].plot(ema.index, ema.values, color="tab:purple", lw=0.9, label="200日EMA")
    axes[0].set_yscale("log"); axes[0].set_ylabel("價格")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_title("曝險走勢對比：基準 vs EMA 建倉濾鏡")
    for axx, (name, _) in zip(axes[1:], VARIANTS.items()):
        _, df = full_store[name]
        axx.fill_between(df.index, df["exposure"],
                         color=COLORS[name], alpha=0.55)
        axx.axhline(1.0, color="gray", lw=0.5)
        axx.set_ylim(0, 1.05); axx.set_ylabel(name, fontsize=8)
        axx.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "ema_exposure.png", dpi=130)

    # 圖3：法幣淨值（全期間）
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, _ in VARIANTS.items():
        _, df = full_store[name]
        ax.plot(df.index, df["value"], color=COLORS[name], lw=1.1, label=name)
    ax.plot(dca_curve.index, dca_curve.values, color=COLORS["DCA"], lw=1.2,
            alpha=0.85, label="DCA")
    cidx = pd.date_range(dca_curve.index[0], periods=nm_full, freq="MS")
    ax.plot(cidx, np.arange(1, nm_full + 1) * CONTRIB, "k--", lw=1, label="累計投入")
    ax.set_yscale("log"); ax.set_ylabel("淨值 (USDT)")
    ax.set_title("法幣淨值：EMA 建倉濾鏡變體 vs DCA（對數刻度）")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "ema_value.png", dpi=130)

    print("\n輸出：results/exp06_ema_rolling/")


if __name__ == "__main__":
    main()
