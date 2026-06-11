"""exp18：想法一定稿版——200EMA regime 切換雙輪（BTC + ETH）。

規格（使用者 2026-06-11 拍板）：
- 遲滯：兩種都跑——price（EMA±3% 突破即換）vs time（連續 7 個交易日
  開盤在另一側才換框架；等待確認期間原 regime 照常運作）。
- 牛市：月投 100% 市價買入 + 止盈寶（call 10%、3% 價外，exp14 驗證值）；
  履約款**留池**作熊市彈藥（不回購）。
- 熊市：月投進池不買；抄底寶 put 比例 sweep {10,20,30}%、
  履約價距離 sweep {1,2,3}%、曝險達 {80,90}% 即停開新倉。
- 觀察項：100% 牛市買入下的曝險路徑（決策 3 的依據）。
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
from ema_accum import build_ema
from ema_wheel import run_ema_wheel
from fetch_binance import load_klines
from fetch_deribit import load_dvol
from grid_vs_dca import CONTRIB, iter_months, window_metrics

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False
RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp18_idea1"
RESULTS.mkdir(parents=True, exist_ok=True)

MAIN_START = "2023-07-01"
CALL_FRAC, CALL_OTM = 0.10, 0.03
HYSTS = ["price", "time"]
PUT_FRACS = [0.10, 0.20, 0.30]
PUT_OTMS = [0.01, 0.02, 0.03]
CAPS = [0.80, 0.90]


def sweep_asset(tag: str, symbol: str, dvol_ccy: str):
    bars = load_klines(symbol, "1h", "2018-01-01", "2026-06-01")
    daily = load_klines(symbol, "1d", "2017-08-17", "2026-06-01")
    ema = build_ema(daily, bars.index)
    iv = load_dvol(dvol_ccy, "2021-03-01", "2026-06-11")["close"].shift(1)

    sub = bars.loc[MAIN_START:]
    nm = sum(1 for _ in iter_months(sub))
    _, ddf = run_dca_cc(sub, 0.0)
    dmet = window_metrics(ddf["value"], nm, CONTRIB)
    dqty = float(ddf["btc"].iloc[-1])
    print(f"\n===== {tag} 主視窗（{nm} 個月）=====")
    print(f"  DCA: MOIC {dmet['moic']:.3f}, IRR {dmet['irr_ann']:+.1%}, "
          f"MDD {dmet['max_dd']:.0%}")

    rows, store = [], {}
    for hyst in HYSTS:
        for pf in PUT_FRACS:
            for po in PUT_OTMS:
                for cap in CAPS:
                    diag, df = run_ema_wheel(
                        sub, ema, iv, CALL_FRAC, otm=CALL_OTM,
                        put_frac=pf, put_otm=po, put_cap=cap,
                        hysteresis=hyst)
                    met = window_metrics(df["value"], nm, CONTRIB)
                    rows.append({
                        "hyst": hyst, "put_frac": pf, "put_otm": po,
                        "cap": cap, **{k: met[k] for k in
                                       ("moic", "irr_ann", "max_dd")},
                        "calmar": met["irr_ann"] / abs(met["max_dd"]),
                        "coin_ratio": diag["final_btc"] / dqty,
                        "expo_mean": diag["expo_mean"],
                        "expo_final": float(df["exposure"].iloc[-1]),
                        "bear_time": diag["bear_time"],
                        "prem": diag["prem_call"] + diag["prem_put"]})
                    store[(hyst, pf, po, cap)] = df
    sdf = pd.DataFrame(rows)
    sdf.insert(0, "asset", tag)
    sdf.to_csv(RESULTS / f"{tag}_sweep.csv", index=False)

    print("  各遲滯最佳（依 MOIC）：")
    best = {}
    for hyst in HYSTS:
        b = sdf[sdf["hyst"] == hyst].sort_values("moic").iloc[-1]
        best[hyst] = b
        print(f"   [{hyst}] put {b['put_frac']:.0%}/{b['put_otm']:.0%}/"
              f"閘{b['cap']:.0%}: MOIC {b['moic']:.3f}, "
              f"IRR {b['irr_ann']:+.1%}, MDD {b['max_dd']:.0%}, "
              f"幣量 {b['coin_ratio']:.0%}, 曝險均值 {b['expo_mean']:.0%}"
              f"/期末 {b['expo_final']:.0%}, 熊態 {b['bear_time']:.0%}")
    print(f"  決策3觀察（100% 牛市買入）：曝險均值範圍 "
          f"{sdf['expo_mean'].min():.0%}~{sdf['expo_mean'].max():.0%}，"
          f"期末範圍 {sdf['expo_final'].min():.0%}~{sdf['expo_final'].max():.0%}")

    # 滾動（各遲滯最佳 vs DCA）
    starts = pd.date_range("2021-06-01", "2024-06-01", freq="3MS", tz="UTC")
    roll = []
    for st in starts:
        s2 = bars.loc[st:st + pd.DateOffset(months=24)]
        nm2 = sum(1 for _ in iter_months(s2))
        if nm2 < 24:
            continue
        _, d2 = run_dca_cc(s2, 0.0)
        m2 = window_metrics(d2["value"], nm2, CONTRIB)
        dq2 = float(d2["btc"].iloc[-1])
        r = {"start": st, "dca_moic": m2["moic"], "dca_mdd": m2["max_dd"]}
        for hyst in HYSTS:
            b = best[hyst]
            diag, df = run_ema_wheel(
                s2, ema, iv, CALL_FRAC, otm=CALL_OTM,
                put_frac=float(b["put_frac"]), put_otm=float(b["put_otm"]),
                put_cap=float(b["cap"]), hysteresis=hyst)
            met = window_metrics(df["value"], nm2, CONTRIB)
            r[f"{hyst}_moic"] = met["moic"]
            r[f"{hyst}_mdd"] = met["max_dd"]
            r[f"{hyst}_coin"] = diag["final_btc"] / dq2
        roll.append(r)
    rdf = pd.DataFrame(roll)
    rdf.to_csv(RESULTS / f"{tag}_rolling.csv", index=False)
    for hyst in HYSTS:
        wr = (rdf[f"{hyst}_moic"] > rdf["dca_moic"]).mean()
        bd = (rdf[f"{hyst}_mdd"] > rdf["dca_mdd"]).mean()
        print(f"  滾動[{hyst}]: MOIC贏DCA {wr:.0%} | MDD較淺 {bd:.0%} | "
              f"幣量均值 {rdf[f'{hyst}_coin'].mean():.0%}")

    # 圖：兩遲滯最佳的淨值與曝險
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(ddf["value"].index, ddf["value"].values,
                 color="tab:orange", lw=1.2, label="DCA")
    for hyst, color in [("price", "tab:blue"), ("time", "tab:green")]:
        b = best[hyst]
        df = store[(hyst, b["put_frac"], b["put_otm"], b["cap"])]
        axes[0].plot(df.index, df["value"], color=color, lw=1.1,
                     label=f"雙輪[{hyst}] put{b['put_frac']:.0%}/"
                           f"{b['put_otm']:.0%}/閘{b['cap']:.0%}")
        axes[1].plot(df.index, df["exposure"], color=color, lw=0.9,
                     label=f"[{hyst}] 曝險")
    cidx = pd.date_range(sub.index[0], periods=nm, freq="MS")
    axes[0].plot(cidx, np.arange(1, nm + 1) * CONTRIB, "k--", lw=1,
                 label="累計投入")
    axes[0].set_title(f"exp18 {tag}：regime 雙輪（{MAIN_START}~）")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / f"{tag}_overview.png", dpi=130)
    return sdf, rdf


def main():
    sweep_asset("BTC", "BTCUSDT", "BTC")
    sweep_asset("ETH", "ETHUSDT", "ETH")
    print("\n輸出：results/exp18_idea1/")


if __name__ == "__main__":
    main()
