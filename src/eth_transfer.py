"""exp17：ETH 移植測試——零漂移樣本下的策略排名重排檢驗。

主視窗（2023-07 起 35 個月）ETH 僅 +3.8%（同期 BTC +146%），
直接檢驗本研究核心預測：「DCA 的王座建立在正漂移上；漂移歸零後
收租類策略相對排名上升、網格不再付踏空稅」。

策略集（全部沿用 BTC 版已驗證參數，僅換資料）：
- 純現金 5% / DCA
- 屯幣網格基準（±15%, ratio=50%, s=3%, 立即重開）
- 純掛買單階梯（D=10%, 月底補成交）
- DCA+CC（exp15-C：25%、EMA上、otm3%、履約即回購；ETH DVOL 定價）
- 抄底寶 DCA（otm=2%, 全池賣 put）
- EMA 方向雙輪（frac=10%）
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
from buy_ladder import run_ladder
from cc_tuning import run_cc_plus
from coin_accum import run_accum
from comparison_report import cash_baseline
from dca_covered_call import run_dca_cc
from dca_via_puts import run_dca_via_puts
from ema_accum import build_ema
from ema_wheel import run_ema_wheel
from fetch_binance import load_klines
from fetch_deribit import load_dvol
from grid_vs_dca import CONTRIB, iter_months, window_metrics

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False
RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp17_eth"
RESULTS.mkdir(parents=True, exist_ok=True)

MAIN_START = "2023-07-01"


def main():
    bars = load_klines("ETHUSDT", "1h", "2018-01-01", "2026-06-01")
    daily = load_klines("ETHUSDT", "1d", "2017-08-17", "2026-06-01")
    ema = build_ema(daily, bars.index)
    iv = load_dvol("ETH", "2021-03-01", "2026-06-11")["close"].shift(1)

    sub = bars.loc[MAIN_START:]
    nm = sum(1 for _ in iter_months(sub))
    px0 = float(sub["open"].iloc[0])
    px1 = float(sub["close"].iloc[-1])
    print(f"=== ETH 主視窗 {MAIN_START}~（{nm} 個月）："
          f"{px0:,.0f} → {px1:,.0f}（{px1/px0-1:+.1%}，近零漂移）===")

    rows, store = [], {}

    final_cash, _ = cash_baseline(sub)
    rows.append({"strategy": "純現金 5%", "moic": final_cash / (nm * CONTRIB),
                 "irr_ann": window_metrics(
                     pd.Series([final_cash], index=[sub.index[-1]]),
                     nm, CONTRIB)["irr_ann"],
                 "max_dd": 0.0, "eth_ratio": 0.0})

    _, ddf = run_dca_cc(sub, 0.0)
    dmet = window_metrics(ddf["value"], nm, CONTRIB)
    dqty = float(ddf["btc"].iloc[-1])
    rows.append({"strategy": "DCA", **{k: dmet[k] for k in
                 ("moic", "irr_ann", "max_dd")}, "eth_ratio": 1.0})
    store["DCA"] = ddf["value"]

    diag, df = run_accum(sub, 0.50, 0.03)
    met = window_metrics(df["value"], nm, CONTRIB)
    rows.append({"strategy": "屯幣網格基準", **{k: met[k] for k in
                 ("moic", "irr_ann", "max_dd")},
                 "eth_ratio": diag["final_btc"] / dqty,
                 "avg_cost_ratio": diag["bucket_avg_cost"] / (nm * CONTRIB / dqty)})
    store["屯幣網格"] = df["value"]

    diag, df = run_ladder(sub, 0.10, "fill_eom")
    met = window_metrics(df["value"], nm, CONTRIB)
    rows.append({"strategy": "買單階梯 D=10% 月底補", **{k: met[k] for k in
                 ("moic", "irr_ann", "max_dd")},
                 "eth_ratio": diag["final_btc"] / dqty,
                 "avg_cost_ratio": diag["avg_cost"] / (nm * CONTRIB / dqty)})

    diag, df = run_cc_plus(sub, ema, iv, inst_rebuy=True)
    met = window_metrics(df["value"], nm, CONTRIB)
    rows.append({"strategy": "DCA+CC(exp15-C)", **{k: met[k] for k in
                 ("moic", "irr_ann", "max_dd")},
                 "eth_ratio": diag["final_btc"] / dqty,
                 "premium": diag["prem_call"]})
    store["DCA+CC"] = df["value"]

    diag, df = run_dca_via_puts(sub, iv, 0.02)
    met = window_metrics(df["value"], nm, CONTRIB)
    rows.append({"strategy": "抄底寶DCA otm=2%", **{k: met[k] for k in
                 ("moic", "irr_ann", "max_dd")},
                 "eth_ratio": diag["final_btc"] / dqty,
                 "premium": diag["premium"]})
    store["抄底寶DCA"] = df["value"]

    diag, df = run_ema_wheel(sub, ema, iv, 0.10)
    met = window_metrics(df["value"], nm, CONTRIB)
    rows.append({"strategy": "EMA雙輪 frac=10%", **{k: met[k] for k in
                 ("moic", "irr_ann", "max_dd")},
                 "eth_ratio": diag["final_btc"] / dqty,
                 "premium": diag["prem_call"] + diag["prem_put"]})
    store["EMA雙輪"] = df["value"]

    mdf = pd.DataFrame(rows)
    mdf["calmar"] = mdf["irr_ann"] / mdf["max_dd"].abs().replace(0, np.nan)
    mdf.to_csv(RESULTS / "main_window.csv", index=False)
    pd.set_option("display.width", 200)
    print(mdf[["strategy", "moic", "irr_ann", "max_dd", "calmar",
               "eth_ratio"]].round(3).to_string(index=False))

    # ===== 滾動 24 個月（核心四策略）=====
    starts = pd.date_range("2021-06-01", "2024-06-01", freq="3MS", tz="UTC")
    roll = []
    for st in starts:
        s2 = bars.loc[st:st + pd.DateOffset(months=24)]
        nm2 = sum(1 for _ in iter_months(s2))
        if nm2 < 24:
            continue
        r = {"start": st}
        _, d2 = run_dca_cc(s2, 0.0)
        m2 = window_metrics(d2["value"], nm2, CONTRIB)
        dq2 = float(d2["btc"].iloc[-1])
        r["dca_moic"], r["dca_mdd"] = m2["moic"], m2["max_dd"]
        dg, dfg = run_accum(s2, 0.50, 0.03)
        mg = window_metrics(dfg["value"], nm2, CONTRIB)
        r["grid_moic"], r["grid_mdd"] = mg["moic"], mg["max_dd"]
        r["grid_eth"] = dg["final_btc"] / dq2
        dc_, dfc = run_cc_plus(s2, ema, iv, inst_rebuy=True)
        mc = window_metrics(dfc["value"], nm2, CONTRIB)
        r["cc_moic"], r["cc_mdd"] = mc["moic"], mc["max_dd"]
        r["cc_eth"] = dc_["final_btc"] / dq2
        dp, dfp = run_dca_via_puts(s2, iv, 0.02)
        mp = window_metrics(dfp["value"], nm2, CONTRIB)
        r["put_moic"], r["put_mdd"] = mp["moic"], mp["max_dd"]
        r["put_eth"] = dp["final_btc"] / dq2
        roll.append(r)
    rdf = pd.DataFrame(roll)
    rdf.to_csv(RESULTS / "rolling24m.csv", index=False)
    print("\n=== ETH 滾動 24 個月 ===")
    print(rdf.assign(start=rdf["start"].dt.strftime("%Y-%m"))[
        ["start", "dca_moic", "grid_moic", "cc_moic", "put_moic",
         "grid_eth", "cc_eth", "put_eth"]].round(3).to_string(index=False))
    for tag, name in [("grid", "屯幣網格"), ("cc", "DCA+CC"), ("put", "抄底寶DCA")]:
        wr = (rdf[f"{tag}_moic"] > rdf["dca_moic"]).mean()
        bd = (rdf[f"{tag}_mdd"] > rdf["dca_mdd"]).mean()
        print(f"  {name}: MOIC贏DCA {wr:.0%} | MDD較淺 {bd:.0%} | "
              f"幣量均值 {rdf[f'{tag}_eth'].mean():.0%}")

    # ===== 圖 =====
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
    COLORS = {"DCA": "tab:orange", "屯幣網格": "tab:gray",
              "DCA+CC": "tab:green", "抄底寶DCA": "tab:blue",
              "EMA雙輪": "tab:purple"}
    for name, curve in store.items():
        axes[0].plot(curve.index, curve.values, color=COLORS[name],
                     lw=1.1, label=name)
    cidx = pd.date_range(store["DCA"].index[0], periods=nm, freq="MS")
    axes[0].plot(cidx, np.arange(1, nm + 1) * CONTRIB, "k--", lw=1,
                 label="累計投入")
    axes[0].set_title(f"ETH 主視窗 {MAIN_START}~（總漲幅 {px1/px0-1:+.1%}，"
                      "近零漂移）：法幣淨值")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].plot(rdf["start"], rdf["dca_moic"], "-o", ms=3,
                 color="tab:orange", label="DCA")
    axes[1].plot(rdf["start"], rdf["grid_moic"], "-o", ms=3,
                 color="tab:gray", label="屯幣網格")
    axes[1].plot(rdf["start"], rdf["cc_moic"], "-o", ms=3,
                 color="tab:green", label="DCA+CC")
    axes[1].plot(rdf["start"], rdf["put_moic"], "-o", ms=3,
                 color="tab:blue", label="抄底寶DCA")
    axes[1].axhline(1, color="gray", lw=0.8)
    axes[1].set_title("ETH 滾動 24 個月 MOIC")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "eth_overview.png", dpi=130)
    print("\n輸出：results/exp17_eth/")


if __name__ == "__main__":
    main()
