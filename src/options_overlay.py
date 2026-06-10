"""exp09：選擇權層（第二部分）— 幣桶 25% 賣 1 天期 covered call（止盈寶機制）。

定價：Black-Scholes + Deribit DVOL（真實市場 IV，前日收盤，防前視），
      iv_mult 敏感度另列。30 天期 DVOL 用於 1 天期合約的期限結構誤差列為限制。
合約：1 天到期、價外 {2%, 3%, 5%} sweep、每 UTC 日輪動。
觸發：常態輪動 vs 僅價格 > 200EMA（原始構想的止盈寶開關）。
基底：第一部分基準屯幣網格（ratio=50%, s=3%, 立即重開），不賣 put。
主視窗 2023-07 起；滾動 24 個月視窗起點 2021-06 起（受 DVOL 起始限制）。
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
from fetch_deribit import load_dvol
from grid_vs_dca import CONTRIB, iter_months, run_dca, window_metrics

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp09_covered_call"
RESULTS.mkdir(parents=True, exist_ok=True)

RATIO, S = 0.50, 0.03
CALL_FRAC = 0.25
OTM_SWEEP = [0.02, 0.03, 0.05]
MAIN_START = "2023-07-01"


def eval_one(sub, ema, iv, otm=None, trigger="always"):
    nm = sum(1 for _ in iter_months(sub))
    kw = {}
    if otm is not None:
        kw = dict(call_frac=CALL_FRAC, call_otm=otm, call_trigger=trigger, iv=iv)
    diag, df = run_accum(sub, RATIO, S, ema=ema, **kw)
    met = window_metrics(df["value"], nm, CONTRIB)
    return diag, df, met, nm


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")
    daily = load_klines("BTCUSDT", "1d", "2017-08-17", "2026-06-01")
    ema = build_ema(daily, bars.index)
    dvol = load_dvol()
    iv = dvol["close"].shift(1)          # 前日收盤 IV，防前視

    # ===== (1) 主視窗 =====
    main_bars = bars.loc[MAIN_START:]
    nm = sum(1 for _ in iter_months(main_bars))
    dm, dc = run_dca(main_bars)
    dqty = float(dm["qty_cum"].iloc[-1])
    dmet = window_metrics(dc, nm, CONTRIB)
    print(f"=== 主視窗 {MAIN_START}~（{nm} 個月）===")
    print(f"  DCA: MOIC {dmet['moic']:.3f}, MDD {dmet['max_dd']:.0%}, BTC {dqty:.4f}")

    rows, store = [], {}
    configs = [("基準(無call)", None, "always")]
    configs += [(f"常態輪動 otm={o:.0%}", o, "always") for o in OTM_SWEEP]
    configs += [(f"EMA上才賣 otm={o:.0%}", o, "above_ema") for o in OTM_SWEEP]
    for name, otm, trig in configs:
        diag, df, met, _ = eval_one(main_bars, ema, iv, otm, trig)
        store[name] = df
        ex_rate = diag["n_exercised"] / max(diag["n_calls"], 1)
        rows.append({"config": name, "otm": otm, "trigger": trig,
                     **{k: diag[k] for k in
                        ["final_btc", "bucket_avg_cost", "avg_exposure",
                         "n_calls", "n_exercised", "premium_total",
                         "called_qty", "called_proceeds"]},
                     **met, "btc_ratio": diag["final_btc"] / dqty,
                     "ex_rate": ex_rate})
        extra = ("" if otm is None else
                 f", 賣call {diag['n_calls']} 次/履約率 {ex_rate:.0%}, "
                 f"權利金 {diag['premium_total']:,.0f}, "
                 f"被叫走 {diag['called_qty']:.3f} BTC")
        print(f"  {name}: MOIC {met['moic']:.3f}, MDD {met['max_dd']:.0%}, "
              f"BTC {diag['final_btc']:.4f} ({diag['final_btc']/dqty:.0%} DCA), "
              f"曝險 {diag['avg_exposure']:.0%}{extra}")
    mdf = pd.DataFrame(rows)
    mdf.to_csv(RESULTS / "main_window.csv", index=False)

    grid_only = mdf[mdf["otm"].notna()]
    best_always = grid_only[grid_only["trigger"] == "always"] \
        .sort_values("moic").iloc[-1]
    best_ema = grid_only[grid_only["trigger"] == "above_ema"] \
        .sort_values("moic").iloc[-1]
    print(f"→ 最佳常態輪動：otm={best_always['otm']:.0%} | "
          f"最佳EMA觸發：otm={best_ema['otm']:.0%}")

    # IV 敏感度（最佳常態輪動，iv_mult=0.8）
    diag8, df8, met8, _ = eval_one(main_bars, ema, iv, float(best_always["otm"]))
    diag_s, df_s = run_accum(main_bars, RATIO, S, ema=ema, call_frac=CALL_FRAC,
                             call_otm=float(best_always["otm"]), iv=iv,
                             iv_mult=0.8)
    met_s = window_metrics(df_s["value"], nm, CONTRIB)
    print(f"  IV敏感度 iv_mult=0.8：MOIC {met_s['moic']:.3f} "
          f"(權利金 {diag_s['premium_total']:,.0f})")

    # ===== (2) 滾動 24 個月（受 DVOL 限制，2021-06 起）=====
    starts = pd.date_range("2021-06-01", "2024-06-01", freq="3MS", tz="UTC")
    roll = []
    for st in starts:
        en = st + pd.DateOffset(months=24)
        sub = bars.loc[st:en]
        if sum(1 for _ in iter_months(sub)) < 24:
            continue
        dm_, dc_ = run_dca(sub)
        dqty_ = float(dm_["qty_cum"].iloc[-1])
        dmet_ = window_metrics(dc_, sum(1 for _ in iter_months(sub)), CONTRIB)
        r = {"start": st, "dca_moic": dmet_["moic"], "dca_mdd": dmet_["max_dd"]}
        for tag, otm, trig in [
                ("base", None, "always"),
                ("always", float(best_always["otm"]), "always"),
                ("ematrig", float(best_ema["otm"]), "above_ema")]:
            diag, df, met, _ = eval_one(sub, ema, iv, otm, trig)
            r[f"{tag}_moic"] = met["moic"]
            r[f"{tag}_mdd"] = met["max_dd"]
            r[f"{tag}_btc_ratio"] = diag["final_btc"] / dqty_
            r[f"{tag}_prem"] = diag["premium_total"]
        roll.append(r)
    rdf = pd.DataFrame(roll)
    rdf.to_csv(RESULTS / "rolling24m.csv", index=False)
    print("\n=== 滾動視窗（24 個月）===")
    print(rdf.assign(start=rdf["start"].dt.strftime("%Y-%m"))[
        ["start", "dca_moic", "base_moic", "always_moic", "ematrig_moic",
         "base_btc_ratio", "always_btc_ratio", "ematrig_btc_ratio"]]
        .round(3).to_string(index=False))

    # ===== 圖表 =====
    SHOW = {"DCA": (dc, "tab:orange"),
            "基準(無call)": (store["基準(無call)"], "tab:gray"),
            f"常態輪動 otm={best_always['otm']:.0%}":
                (store[f"常態輪動 otm={best_always['otm']:.0%}"], "tab:green"),
            f"EMA上才賣 otm={best_ema['otm']:.0%}":
                (store[f"EMA上才賣 otm={best_ema['otm']:.0%}"], "tab:blue")}

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    for name, (df, color) in SHOW.items():
        y = df if name == "DCA" else df["value"]
        ax.plot(y.index, y.values, color=color, lw=1.2, label=name)
    cidx = pd.date_range(dc.index[0], periods=nm, freq="MS")
    ax.plot(cidx, np.arange(1, nm + 1) * CONTRIB, "k--", lw=1, label="累計投入")
    ax.set_title(f"主視窗 {MAIN_START}~：covered call 層（幣桶 25%, 1天期）法幣淨值")
    ax.set_ylabel("USDT"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    dq = dc / main_bars["close"].reindex(dc.index)
    ax.plot(dq.index, dq.values, color="tab:orange", lw=1.2, label="DCA")
    for name, (df, color) in SHOW.items():
        if name == "DCA":
            continue
        ax.plot(df.index, df["btc"], color=color, lw=1.1, label=name)
    ax.set_title("累積持幣量 (BTC)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    for name, (df, color) in SHOW.items():
        if name == "DCA":
            continue
        ax.plot(df.index, df["prem_cum"], color=color, lw=1.1, label=name)
    ax.set_title("累計權利金收入 (USDT)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    for name, (df, color) in SHOW.items():
        if name == "DCA":
            continue
        ax.plot(df.index, df["exposure"], color=color, lw=1.0, label=name)
    ax.axhline(1.0, color="tab:orange", lw=1.2, label="DCA = 1")
    ax.set_ylim(0, 1.05); ax.set_title("曝險走勢")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "main_overview.png", dpi=130)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    ax.plot(rdf["start"], rdf["dca_moic"], "-o", ms=3, color="tab:orange", label="DCA")
    for tag, color, lbl in [("base", "tab:gray", "基準"),
                            ("always", "tab:green", "常態輪動"),
                            ("ematrig", "tab:blue", "EMA觸發")]:
        ax.plot(rdf["start"], rdf[f"{tag}_moic"], "-o", ms=3, color=color, label=lbl)
    ax.axhline(1, color="gray", lw=0.8)
    ax.set_title("滾動 24 個月 MOIC"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    for tag, color, lbl in [("base", "tab:gray", "基準"),
                            ("always", "tab:green", "常態輪動"),
                            ("ematrig", "tab:blue", "EMA觸發")]:
        ax.plot(rdf["start"], rdf[f"{tag}_btc_ratio"], "-o", ms=3,
                color=color, label=lbl)
    ax.axhline(1, color="tab:orange", lw=1.2, label="DCA = 1")
    ax.set_title("期末持幣量 / DCA"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[2]
    for tag, color, lbl in [("always", "tab:green", "常態輪動"),
                            ("ematrig", "tab:blue", "EMA觸發")]:
        ax.plot(rdf["start"], rdf[f"{tag}_prem"], "-o", ms=3, color=color, label=lbl)
    ax.set_title("視窗內權利金收入 (USDT)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for axx in axes:
        axx.set_xlabel("視窗起始月")
    fig.suptitle("exp09：covered call 層滾動視窗檢驗", fontsize=13)
    fig.tight_layout(); fig.savefig(RESULTS / "rolling.png", dpi=130)

    print("\n輸出：results/exp09_covered_call/")


if __name__ == "__main__":
    main()
