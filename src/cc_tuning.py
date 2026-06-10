"""exp15：exp10 的消融式微調 —— 降讓渡、提毛收、加第二收入源。

變體（在 exp10 = DCA + CC25% EMA上 otm3% 月度全回購 的基礎上）：
  A baseline：exp10 原版
  B vol_strike：履約價改為 spot×(1 + k×σ_1d)，k=1.25（恆定 delta 賣法）
  C inst_rebuy：call 履約款當天開盤立刻回購（≈現金結算，幣量不流失）
  D bear_put：EMA−3% 之下月投不市價買，現金池每日 25% 賣 put 接貨（exp14 融合）
  E all：B+C+D

每變體輸出毛權利金 / 履約讓渡 / 淨效果的分解。
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
from dual_wheel import bs_put
from ema_accum import build_ema
from fetch_binance import load_klines
from fetch_deribit import load_dvol
from grid_vs_dca import CONTRIB, FEE, iter_months, run_dca, window_metrics

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp15_cc_tuning"
RESULTS.mkdir(parents=True, exist_ok=True)

MAIN_START = "2023-07-01"
FRAC = 0.25
OTM_FIX = 0.03
K_SIGMA = 1.25
BUFFER = 0.03


def run_cc_plus(bars, ema, iv, vol_strike=False, inst_rebuy=False,
                bear_put=False, frac=FRAC, contrib=CONTRIB):
    pool = qty = reserved = 0.0
    live_call = live_put = None
    cur_day = None
    state = "bull"
    n_c = n_p = ex_c = ex_p = 0
    prem_c = prem_p = giveback_c = 0.0
    vals, btcs, prems, idx = [], [], [], []

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
                e = float(ema.loc[ts])
                if np.isfinite(e):
                    if op > e * (1 + BUFFER):
                        state = "bull"
                    elif op < e * (1 - BUFFER):
                        state = "bear"
                if live_call is not None:
                    qc, kk = live_call
                    if op > kk and qty > 0:
                        qc = min(qc, qty)
                        qty -= qc
                        pool += qc * kk
                        giveback_c += qc * (op - kk)   # 讓渡 = 結算價−履約價
                        ex_c += 1
                        if inst_rebuy:                  # 即時回購
                            spend = qc * kk
                            qty += spend * (1 - FEE) / op
                            pool -= spend
                    live_call = None
                if live_put is not None:
                    coll, kk = live_put
                    reserved -= coll
                    if op < kk:
                        pool -= coll
                        qty += coll / kk
                        ex_p += 1
                    live_put = None
            if month_buy:
                if (not bear_put) or state == "bull":   # 市價月投（全池回購）
                    buy = max(pool - reserved, 0.0)
                    if buy > 1e-9:
                        qty += buy * (1 - FEE) / op
                        pool -= buy
                month_buy = False                       # bear+bear_put：留池賣put
            if roll:
                sigma = iv.asof(ts)
                if np.isfinite(sigma):
                    sig = float(sigma) / 100
                    otm = K_SIGMA * sig / np.sqrt(365) if vol_strike else OTM_FIX
                    e = float(ema.loc[ts])
                    if state == "bull" and np.isfinite(e) and op > e \
                            and qty > 0 and live_call is None:
                        qc = frac * qty
                        kk = op * (1 + otm)
                        prem = bs_call(op, kk, sig, 1 / 365) * qc
                        pool += prem
                        prem_c += prem
                        live_call = (qc, kk)
                        n_c += 1
                    elif bear_put and state == "bear" and live_put is None \
                            and (pool - reserved) > 1.0:
                        coll = frac * (pool - reserved)
                        kk = op * (1 - otm)
                        prem = bs_put(op, kk, sig, 1 / 365) * (coll / kk)
                        pool += prem
                        prem_p += prem
                        live_put = (coll, kk)
                        reserved += coll
                        n_p += 1
            v = pool + qty * c
            vals.append(v)
            btcs.append(qty)
            prems.append(prem_c + prem_p)

    df = pd.DataFrame({"value": vals, "btc": btcs, "prem_cum": prems},
                      index=pd.DatetimeIndex(idx))
    diag = {"final_btc": qty, "n_calls": n_c, "ex_calls": ex_c,
            "n_puts": n_p, "ex_puts": ex_p,
            "prem_call": prem_c, "prem_put": prem_p,
            "giveback_call": giveback_c}
    return diag, df


VARIANTS = {
    "A 基準(exp10)": dict(),
    "B vol履約價": dict(vol_strike=True),
    "C 即時回購": dict(inst_rebuy=True),
    "D 熊態put": dict(bear_put=True),
    "E 全合併": dict(vol_strike=True, inst_rebuy=True, bear_put=True),
}


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")
    daily = load_klines("BTCUSDT", "1d", "2017-08-17", "2026-06-01")
    ema = build_ema(daily, bars.index)
    iv = load_dvol()["close"].shift(1)

    sub = bars.loc[MAIN_START:]
    nm = sum(1 for _ in iter_months(sub))
    dm, dc = run_dca(sub)
    dqty = float(dm["qty_cum"].iloc[-1])
    dmet = window_metrics(dc, nm, CONTRIB)
    dca_final = float(dc.iloc[-1])
    print(f"=== 主視窗 {MAIN_START}~（{nm} 個月）===")
    print(f"  純 DCA: MOIC {dmet['moic']:.3f}, IRR {dmet['irr_ann']:+.1%}, "
          f"MDD {dmet['max_dd']:.0%}, BTC {dqty:.4f}")

    rows, store = [], {}
    for name, kw in VARIANTS.items():
        diag, df = run_cc_plus(sub, ema, iv, **kw)
        met = window_metrics(df["value"], nm, CONTRIB)
        store[name] = df
        net_edge = float(df["value"].iloc[-1]) - dca_final
        rows.append({"variant": name, **diag, **met,
                     "btc_ratio": diag["final_btc"] / dqty,
                     "net_edge_vs_dca": net_edge})
        print(f"  {name}: MOIC {met['moic']:.3f}, IRR {met['irr_ann']:+.1%}, "
              f"MDD {met['max_dd']:.0%}, BTC {diag['final_btc']/dqty:.0%} | "
              f"毛權利金 C{diag['prem_call']:,.0f}+P{diag['prem_put']:,.0f}, "
              f"call讓渡 {diag['giveback_call']:,.0f}, "
              f"淨優勢(終值-DCA) {net_edge:+,.0f} | "
              f"call {diag['n_calls']}/{diag['ex_calls']}, "
              f"put {diag['n_puts']}/{diag['ex_puts']}")
    pd.DataFrame(rows).to_csv(RESULTS / "main_window.csv", index=False)

    # ===== 滾動：A、最佳單改、E =====
    best_single = max(rows[1:4], key=lambda r: r["net_edge_vs_dca"])["variant"]
    print(f"\n→ 最佳單一改動：{best_single}")
    sel = ["A 基準(exp10)", best_single, "E 全合併"]
    starts = pd.date_range("2021-06-01", "2024-06-01", freq="3MS", tz="UTC")
    roll = []
    for st in starts:
        s2 = bars.loc[st:st + pd.DateOffset(months=24)]
        nm2 = sum(1 for _ in iter_months(s2))
        if nm2 < 24:
            continue
        dm2, dc2 = run_dca(s2)
        dq2 = float(dm2["qty_cum"].iloc[-1])
        dmet2 = window_metrics(dc2, nm2, CONTRIB)
        r = {"start": st, "dca_moic": dmet2["moic"], "dca_mdd": dmet2["max_dd"]}
        for name in sel:
            diag, df = run_cc_plus(s2, ema, iv, **VARIANTS[name])
            met = window_metrics(df["value"], nm2, CONTRIB)
            tag = name.split()[0]
            r[f"{tag}_moic"] = met["moic"]
            r[f"{tag}_mdd"] = met["max_dd"]
            r[f"{tag}_btc"] = diag["final_btc"] / dq2
        roll.append(r)
    rdf = pd.DataFrame(roll)
    rdf.to_csv(RESULTS / "rolling24m.csv", index=False)
    print("\n=== 滾動 24 個月 ===")
    cols = ["start", "dca_moic"] + [f"{n.split()[0]}_{m}" for n in sel
                                    for m in ("moic", "btc")]
    print(rdf.assign(start=rdf["start"].dt.strftime("%Y-%m"))[cols]
          .round(3).to_string(index=False))
    for name in sel:
        tag = name.split()[0]
        wr = (rdf[f"{tag}_moic"] > rdf["dca_moic"]).mean()
        bd = (rdf[f"{tag}_mdd"] > rdf["dca_mdd"]).mean()
        print(f"  {name}: MOIC贏DCA {wr:.0%} | MDD較淺 {bd:.0%} | "
              f"幣量均值 {rdf[f'{tag}_btc'].mean():.0%}")

    # 圖
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(dc.index, dc.values, color="tab:orange", lw=1.3, label="純 DCA")
    CB = {"A 基準(exp10)": "tab:gray", "B vol履約價": "tab:blue",
          "C 即時回購": "tab:green", "D 熊態put": "tab:purple",
          "E 全合併": "tab:red"}
    for name, df in store.items():
        axes[0].plot(df.index, df["value"], color=CB[name], lw=1.0, label=name)
    axes[0].set_title(f"exp15 消融（{MAIN_START}~）：法幣淨值")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    for name, df in store.items():
        axes[1].plot(df.index, df["btc"], color=CB[name], lw=1.0, label=name)
    dq_curve = dc / sub["close"].reindex(dc.index)
    axes[1].plot(dq_curve.index, dq_curve.values, color="tab:orange", lw=1.3,
                 label="純 DCA")
    axes[1].set_title("累積持幣量 (BTC)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "main_overview.png", dpi=130)
    print("\n輸出：results/exp15_cc_tuning/")


if __name__ == "__main__":
    main()
