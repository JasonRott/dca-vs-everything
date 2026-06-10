"""exp13：雙輪穩態 —— DCA 基底 + 曝險帶比例控制的抄底寶/止盈寶。

機制：
- 月投 1000 進池；月初「帶感知買入」：只把曝險買到帶中點（不足才買、
  超過不買），其餘留池吃 5% 息——現金流與控制器同向。
- 每 UTC 日（先結算昨日合約，再開新單）：
  - 曝險 > 上界 U：對超額量的 gain=50% 賣 1 天期 otm 3% call（止盈寶）
  - 曝險 < 下界 L：用未保留現金對缺額的 50% 賣 1 天期 otm 3% put（抄底寶，
    現金擔保、開倉即保留）
  - 帶內：不動作
- 權利金、被叫走款項入池；put 履約則保證金換幣。選擇權結算不另計費
  （與 exp09/10 一致）；DVOL 前日收盤 + Black-Scholes 定價。
- 曝險帶 sweep：[50,70] / [60,80] / [70,90]。

穩態指標：帶內時間比例、曝險均值/標準差；傳統指標照舊。
對照：純 DCA、exp10 最佳（DCA+CC 25% EMA上 otm3%）。
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
from coin_accum import HOURLY_R, bs_call
from dca_covered_call import run_dca_cc
from ema_accum import build_ema
from fetch_binance import load_klines
from fetch_deribit import load_dvol
from grid_vs_dca import CONTRIB, FEE, iter_months, window_metrics

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp13_dual_wheel"
RESULTS.mkdir(parents=True, exist_ok=True)

MAIN_START = "2023-07-01"
BANDS = [(0.50, 0.70), (0.60, 0.80), (0.70, 0.90)]
OTM = 0.03
GAIN = 0.5


def bs_put(s, k, sigma, t):
    return bs_call(s, k, sigma, t) - s + k     # r=0 的買賣權平價


def run_dual_wheel(bars: pd.DataFrame, band: tuple[float, float],
                   iv: pd.Series, otm: float = OTM, gain: float = GAIN,
                   contrib: float = CONTRIB, buy_target: str = "floor"):
    """buy_target：
    - "dca"  ：月投固定買入 contrib（真 DCA 流量；call 履約款滯留池中
               形成現金緩衝，選擇權承擔全部再平衡）
    - "floor"：月投只買到帶下緣（曝險貼地、僅 put 微量出單）
    - "mid"  ：月投買到帶中點（流量主導再平衡，選擇權閒置，對照組）"""
    L, U = band
    mid = (L + U) / 2
    tgt = L if buy_target == "floor" else mid
    pool = qty = reserved = 0.0
    live_call = live_put = None        # (數量, K) / (保證金, K)
    n_c = n_p = ex_c = ex_p = 0
    prem_c = prem_p = 0.0
    cur_day = None
    vals, expos, btcs, prems, idx = [], [], [], [], []

    for mstart, mbars in iter_months(bars):
        pool += contrib
        month_buy = True
        idx.extend(mbars.index)
        for ts, o, h, l, c in mbars[["open", "high", "low", "close"]] \
                .itertuples(index=True):
            pool *= (1 + HOURLY_R)
            op = float(o)
            day = ts.normalize()
            roll = day != cur_day
            if roll:
                cur_day = day
                if live_call is not None:               # 結算止盈寶
                    qc, kk = live_call
                    if op > kk and qty > 0:
                        qc = min(qc, qty)
                        qty -= qc
                        pool += qc * kk
                        ex_c += 1
                    live_call = None
                if live_put is not None:                # 結算抄底寶
                    coll, kk = live_put
                    reserved -= coll
                    if op < kk:
                        pool -= coll
                        qty += coll / kk
                        ex_p += 1
                    live_put = None
            if month_buy:                               # 月投買入
                if buy_target == "dca":                 # 固定買 contrib
                    buy = min(contrib, pool - reserved)
                else:                                   # 帶感知：買到目標位
                    tot = pool + qty * op
                    buy = min(max(tgt * tot - qty * op, 0.0), pool - reserved)
                if buy > 1e-9:
                    qty += buy * (1 - FEE) / op
                    pool -= buy
                month_buy = False
            if roll:                                    # 比例控制開單
                tot = pool + qty * op
                expo = qty * op / tot if tot > 0 else 0.0
                sigma = iv.asof(ts)
                if np.isfinite(sigma) and tot > 0:
                    sig = float(sigma) / 100
                    if expo > U and qty > 0:
                        qc = min(gain * (expo - U) * tot / op, qty)
                        kk = op * (1 + otm)
                        prem = bs_call(op, kk, sig, 1 / 365) * qc
                        pool += prem
                        prem_c += prem
                        live_call = (qc, kk)
                        n_c += 1
                    elif expo < L and (pool - reserved) > 1.0:
                        coll = min(gain * (L - expo) * tot, pool - reserved)
                        kk = op * (1 - otm)
                        prem = bs_put(op, kk, sig, 1 / 365) * (coll / kk)
                        pool += prem
                        prem_p += prem
                        live_put = (coll, kk)
                        reserved += coll
                        n_p += 1
            v = pool + qty * c
            vals.append(v)
            expos.append(qty * c / v if v > 0 else 0.0)
            btcs.append(qty)
            prems.append(prem_c + prem_p)

    df = pd.DataFrame({"value": vals, "exposure": expos, "btc": btcs,
                       "prem_cum": prems}, index=pd.DatetimeIndex(idx))
    e = df["exposure"]
    diag = {"band": f"[{L:.0%},{U:.0%}]",
            "final_btc": qty,
            "time_in_band": float(((e >= L) & (e <= U)).mean()),
            "expo_mean": float(e.mean()), "expo_std": float(e.std()),
            "n_calls": n_c, "n_puts": n_p, "ex_calls": ex_c, "ex_puts": ex_p,
            "prem_call": prem_c, "prem_put": prem_p}
    return diag, df


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")
    daily = load_klines("BTCUSDT", "1d", "2017-08-17", "2026-06-01")
    ema = build_ema(daily, bars.index)
    iv = load_dvol()["close"].shift(1)

    sub = bars.loc[MAIN_START:]
    nm = sum(1 for _ in iter_months(sub))

    # 對照組
    _, dca_df = run_dca_cc(sub, 0.0)
    dca_met = window_metrics(dca_df["value"], nm, CONTRIB)
    dqty = float(dca_df["btc"].iloc[-1])
    diag_r, ref_df = run_dca_cc(sub, 0.25, 0.03, "above_ema", ema, iv)
    ref_met = window_metrics(ref_df["value"], nm, CONTRIB)

    print(f"=== 主視窗 {MAIN_START}~（{nm} 個月）===")
    print(f"  純 DCA: MOIC {dca_met['moic']:.3f}, IRR {dca_met['irr_ann']:+.1%}, "
          f"MDD {dca_met['max_dd']:.0%}, BTC {dqty:.4f}")
    print(f"  DCA+CC25%EMA(exp10): MOIC {ref_met['moic']:.3f}, "
          f"IRR {ref_met['irr_ann']:+.1%}, MDD {ref_met['max_dd']:.0%}, "
          f"BTC {diag_r['final_btc']:.4f}")

    rows, store = [], {}
    # 兩個對照變體（帶 [60,80]）：流量再平衡 / 帶下緣錨定
    for vname, bt in [("flow_mid", "mid"), ("wheel_floor", "floor")]:
        diag_f, df_f = run_dual_wheel(sub, (0.60, 0.80), iv, buy_target=bt)
        met_f = window_metrics(df_f["value"], nm, CONTRIB)
        print(f"  {vname}[60,80](對照): MOIC {met_f['moic']:.3f}, "
              f"IRR {met_f['irr_ann']:+.1%}, MDD {met_f['max_dd']:.0%}, "
              f"BTC {diag_f['final_btc']:.4f}, "
              f"權利金 {diag_f['prem_call']+diag_f['prem_put']:,.0f}")
        rows.append({**diag_f, **met_f, "variant": vname,
                     "btc_ratio": diag_f["final_btc"] / dqty,
                     "calmar": met_f["irr_ann"] / abs(met_f["max_dd"])})
    for band in BANDS:
        diag, df = run_dual_wheel(sub, band, iv, buy_target="dca")
        met = window_metrics(df["value"], nm, CONTRIB)
        store[diag["band"]] = df
        rows.append({**diag, **met, "variant": "dca_band",
                     "btc_ratio": diag["final_btc"] / dqty,
                     "calmar": met["irr_ann"] / abs(met["max_dd"])})
        print(f"  帶{diag['band']}: MOIC {met['moic']:.3f}, "
              f"IRR {met['irr_ann']:+.1%}, MDD {met['max_dd']:.0%}, "
              f"Calmar {met['irr_ann']/abs(met['max_dd']):.2f}, "
              f"BTC {diag['final_btc']:.4f} ({diag['final_btc']/dqty:.0%}), "
              f"帶內時間 {diag['time_in_band']:.0%}, "
              f"曝險 {diag['expo_mean']:.0%}±{diag['expo_std']:.0%}, "
              f"call {diag['n_calls']}/{diag['ex_calls']}履約, "
              f"put {diag['n_puts']}/{diag['ex_puts']}履約, "
              f"權利金 C{diag['prem_call']:,.0f}+P{diag['prem_put']:,.0f}")
    pd.DataFrame(rows).to_csv(RESULTS / "main_window.csv", index=False)

    # ===== 滾動 24 個月 =====
    starts = pd.date_range("2021-06-01", "2024-06-01", freq="3MS", tz="UTC")
    roll = []
    for st in starts:
        s2 = bars.loc[st:st + pd.DateOffset(months=24)]
        nm2 = sum(1 for _ in iter_months(s2))
        if nm2 < 24:
            continue
        _, d2 = run_dca_cc(s2, 0.0)
        dmet2 = window_metrics(d2["value"], nm2, CONTRIB)
        dq2 = float(d2["btc"].iloc[-1])
        r = {"start": st, "dca_moic": dmet2["moic"], "dca_mdd": dmet2["max_dd"]}
        for band in BANDS:
            diag, df = run_dual_wheel(s2, band, iv, buy_target="dca")
            met = window_metrics(df["value"], nm2, CONTRIB)
            tag = f"b{int(band[0]*100)}"
            r[f"{tag}_moic"] = met["moic"]
            r[f"{tag}_mdd"] = met["max_dd"]
            r[f"{tag}_btc_ratio"] = diag["final_btc"] / dq2
            r[f"{tag}_inband"] = diag["time_in_band"]
        roll.append(r)
    rdf = pd.DataFrame(roll)
    rdf.to_csv(RESULTS / "rolling24m.csv", index=False)
    print("\n=== 滾動 24 個月 ===")
    print(rdf.assign(start=rdf["start"].dt.strftime("%Y-%m"))[
        ["start", "dca_moic", "b50_moic", "b60_moic", "b70_moic",
         "dca_mdd", "b60_mdd", "b60_btc_ratio", "b60_inband"]]
        .round(3).to_string(index=False))

    # ===== 圖表 =====
    CB = {"[50%,70%]": "tab:blue", "[60%,80%]": "tab:green",
          "[70%,90%]": "tab:purple"}
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 1, 1, 1])
    ax = fig.add_subplot(gs[0])
    ax.plot(dca_df.index, dca_df["value"], color="tab:orange", lw=1.3,
            label="純 DCA")
    ax.plot(ref_df.index, ref_df["value"], color="tab:red", lw=1.0,
            alpha=0.8, label="DCA+CC25%EMA (exp10)")
    for bname, df in store.items():
        ax.plot(df.index, df["value"], color=CB[bname], lw=1.0,
                label=f"雙輪 {bname}")
    cidx = pd.date_range(dca_df.index[0], periods=nm, freq="MS")
    ax.plot(cidx, np.arange(1, nm + 1) * CONTRIB, "k--", lw=1, label="累計投入")
    ax.set_title(f"exp13 雙輪穩態（{MAIN_START}~）：法幣淨值")
    ax.set_ylabel("USDT"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for i, (bname, df) in enumerate(store.items()):
        axx = fig.add_subplot(gs[1 + i])
        Lb, Ub = [float(x.strip("%")) / 100 for x in
                  bname.strip("[]").split(",")]
        axx.axhspan(Lb, Ub, color=CB[bname], alpha=0.12)
        axx.plot(df.index, df["exposure"], color=CB[bname], lw=0.8)
        axx.axhline(Lb, color=CB[bname], lw=0.6, ls="--")
        axx.axhline(Ub, color=CB[bname], lw=0.6, ls="--")
        axx.set_ylim(0, 1.05)
        axx.set_ylabel(f"曝險 {bname}", fontsize=9)
        axx.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "main_overview.png", dpi=130)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(rdf["start"], rdf["dca_moic"], "-o", ms=3,
                 color="tab:orange", label="DCA")
    for band in BANDS:
        tag = f"b{int(band[0]*100)}"
        lbl = f"[{band[0]:.0%},{band[1]:.0%}]"
        axes[0].plot(rdf["start"], rdf[f"{tag}_moic"], "-o", ms=3,
                     color=CB[lbl], label=lbl)
        axes[1].plot(rdf["start"], rdf[f"{tag}_btc_ratio"], "-o", ms=3,
                     color=CB[lbl], label=lbl)
        axes[2].plot(rdf["start"], rdf[f"{tag}_inband"], "-o", ms=3,
                     color=CB[lbl], label=lbl)
    axes[0].axhline(1, color="gray", lw=0.8)
    axes[0].set_title("滾動 24 個月 MOIC")
    axes[1].axhline(1, color="tab:orange", lw=1.2, label="DCA=1")
    axes[1].set_title("期末持幣量 / DCA")
    axes[2].set_title("帶內時間比例")
    for axx in axes:
        axx.set_xlabel("視窗起始月"); axx.legend(fontsize=8); axx.grid(alpha=0.3)
    fig.suptitle("exp13：雙輪穩態滾動視窗", fontsize=13)
    fig.tight_layout(); fig.savefig(RESULTS / "rolling.png", dpi=130)

    print("\n輸出：results/exp13_dual_wheel/")


if __name__ == "__main__":
    main()
