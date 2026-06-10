"""exp10：DCA + 小比例 covered call（不經過網格，直接在 DCA 持倉上收租）。

機制：
- 每月初：現金池（當月投入 + 累積權利金 + 被履約款項，吃 5% 息）全額市價買入 BTC。
- 每 UTC 日：先結算昨日 call（開盤 > 履約價 → 該批幣以履約價賣出入池），
  再以 frac × 當前持幣 賣出 1 天期價外 call（DVOL+BS 定價）。
- 觸發：常態輪動 vs 僅價格 > 200EMA。
- frac sweep {5%, 10%, 25%}（DCA 原生滿曝險 → 用小比例）。

附：主視窗終點敏感度測試（熊市終點對 MOIC 的壓低效果）。
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
from ema_accum import build_ema
from fetch_binance import load_klines
from fetch_deribit import load_dvol
from grid_vs_dca import CONTRIB, FEE, iter_months, window_metrics

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp10_dca_cc"
RESULTS.mkdir(parents=True, exist_ok=True)

MAIN_START = "2023-07-01"
FRAC_SWEEP = [0.05, 0.10, 0.25]
OTM = 0.03


def run_dca_cc(bars: pd.DataFrame, frac: float, otm: float = OTM,
               trigger: str = "always",
               ema: pd.Series | None = None, iv: pd.Series | None = None,
               contrib: float = CONTRIB):
    pool = 0.0
    qty = 0.0
    cur_day = None
    live: tuple[float, float] | None = None
    n_calls = n_ex = 0
    prem_tot = called_qty = called_proceeds = 0.0
    vals, btcs, prems, idx = [], [], [], []

    for mstart, mbars in iter_months(bars):
        pool += contrib
        buy_pending = True
        idx.extend(mbars.index)
        for ts, o, h, l, c in mbars[["open", "high", "low", "close"]] \
                .itertuples(index=True):
            pool *= (1 + HOURLY_R)
            op = float(o)
            day = ts.normalize()
            new_day = frac > 0 and iv is not None and day != cur_day
            if new_day:
                cur_day = day
                if live is not None:                  # 結算昨日批次
                    qc, kk = live
                    if op > kk and qty > 0:
                        qc = min(qc, qty)
                        qty -= qc
                        pool += qc * kk
                        called_qty += qc
                        called_proceeds += qc * kk
                        n_ex += 1
                    live = None
            if buy_pending:                            # 月初全池買入
                qty += pool * (1 - FEE) / op
                pool = 0.0
                buy_pending = False
            if new_day and qty > 0:                    # 賣出新批次
                ok = True
                if trigger == "above_ema" and ema is not None:
                    e = float(ema.loc[ts])
                    ok = np.isfinite(e) and op > e
                if ok:
                    sigma = iv.asof(ts)
                    if np.isfinite(sigma):
                        qc = frac * qty
                        kk = op * (1 + otm)
                        prem = bs_call(op, kk, float(sigma) / 100, 1 / 365) * qc
                        pool += prem
                        prem_tot += prem
                        live = (qc, kk)
                        n_calls += 1
            vals.append(pool + qty * c)
            btcs.append(qty)
            prems.append(prem_tot)

    df = pd.DataFrame({"value": vals, "btc": btcs, "prem_cum": prems},
                      index=pd.DatetimeIndex(idx))
    diag = {"frac": frac, "otm": otm, "trigger": trigger,
            "final_btc": qty, "n_calls": n_calls, "n_exercised": n_ex,
            "premium_total": prem_tot, "called_qty": called_qty,
            "called_proceeds": called_proceeds}
    return diag, df


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")
    daily = load_klines("BTCUSDT", "1d", "2017-08-17", "2026-06-01")
    ema = build_ema(daily, bars.index)
    iv = load_dvol()["close"].shift(1)
    sub = bars.loc[MAIN_START:]
    nm = sum(1 for _ in iter_months(sub))

    # ===== (0) 終點敏感度：熊市終點壓低多少 =====
    print("=== DCA 終點敏感度（起點固定 2023-07）===")
    for end in ["2024-12-31", "2025-06-30", "2025-12-31", "2026-06-01"]:
        s2 = bars.loc[MAIN_START:end]
        nm2 = sum(1 for _ in iter_months(s2))
        _, df0 = run_dca_cc(s2, 0.0)
        met0 = window_metrics(df0["value"], nm2, CONTRIB)
        px = float(s2["close"].iloc[-1])
        print(f"  終點 {end}（{nm2}月, BTC={px:,.0f}）: "
              f"MOIC {met0['moic']:.3f}, IRR {met0['irr_ann']:+.1%}")

    # ===== (1) 主視窗：DCA ± covered call =====
    print(f"\n=== 主視窗 {MAIN_START}~（{nm} 個月）===")
    rows, store = [], {}
    cfgs = [("純 DCA", 0.0, "always")]
    cfgs += [(f"DCA+CC {f:.0%} 常態", f, "always") for f in FRAC_SWEEP]
    cfgs += [(f"DCA+CC {f:.0%} EMA上", f, "above_ema") for f in FRAC_SWEEP]
    dqty = None
    for name, f, trig in cfgs:
        diag, df = run_dca_cc(sub, f, OTM, trig, ema, iv)
        met = window_metrics(df["value"], nm, CONTRIB)
        if dqty is None:
            dqty = diag["final_btc"]
        store[name] = df
        rows.append({"config": name, **diag, **met,
                     "btc_ratio": diag["final_btc"] / dqty,
                     "calmar": met["irr_ann"] / abs(met["max_dd"])})
        ex = diag["n_exercised"] / max(diag["n_calls"], 1)
        extra = ("" if f == 0 else
                 f", 權利金 {diag['premium_total']:,.0f}, 履約率 {ex:.0%}, "
                 f"被叫走 {diag['called_qty']:.3f}")
        print(f"  {name}: MOIC {met['moic']:.3f}, IRR {met['irr_ann']:+.1%}, "
              f"MDD {met['max_dd']:.0%}, Calmar {met['irr_ann']/abs(met['max_dd']):.2f}, "
              f"BTC {diag['final_btc']:.4f} ({diag['final_btc']/dqty:.0%}){extra}")
    mdf = pd.DataFrame(rows)
    mdf.to_csv(RESULTS / "main_window.csv", index=False)

    # otm 敏感度（最佳 frac、兩種觸發）
    best = mdf.iloc[1:].sort_values("calmar").iloc[-1]
    print(f"\n=== otm 敏感度（frac={best['frac']:.0%}, {best['trigger']}）===")
    for otm in [0.02, 0.03, 0.05]:
        diag, df = run_dca_cc(sub, float(best["frac"]), otm,
                              str(best["trigger"]), ema, iv)
        met = window_metrics(df["value"], nm, CONTRIB)
        print(f"  otm={otm:.0%}: MOIC {met['moic']:.3f}, IRR {met['irr_ann']:+.1%}, "
              f"MDD {met['max_dd']:.0%}, BTC比 {diag['final_btc']/dqty:.0%}, "
              f"權利金 {diag['premium_total']:,.0f}")

    # ===== (2) 滾動 24 個月 =====
    starts = pd.date_range("2021-06-01", "2024-06-01", freq="3MS", tz="UTC")
    roll = []
    for st in starts:
        s2 = bars.loc[st:st + pd.DateOffset(months=24)]
        nm2 = sum(1 for _ in iter_months(s2))
        if nm2 < 24:
            continue
        r = {"start": st}
        for tag, f, trig in [("dca", 0.0, "always"),
                             ("cc10", 0.10, "always"),
                             ("cc10ema", 0.10, "above_ema")]:
            diag, df = run_dca_cc(s2, f, OTM, trig, ema, iv)
            met = window_metrics(df["value"], nm2, CONTRIB)
            r[f"{tag}_moic"] = met["moic"]
            r[f"{tag}_mdd"] = met["max_dd"]
            r[f"{tag}_btc"] = diag["final_btc"]
        r["cc10_btc_ratio"] = r["cc10_btc"] / r["dca_btc"]
        r["cc10ema_btc_ratio"] = r["cc10ema_btc"] / r["dca_btc"]
        roll.append(r)
    rdf = pd.DataFrame(roll)
    rdf.to_csv(RESULTS / "rolling24m.csv", index=False)
    print("\n=== 滾動 24 個月（DCA vs DCA+CC10% 兩觸發）===")
    print(rdf.assign(start=rdf["start"].dt.strftime("%Y-%m"))[
        ["start", "dca_moic", "cc10_moic", "cc10ema_moic",
         "dca_mdd", "cc10_mdd", "cc10ema_mdd",
         "cc10_btc_ratio", "cc10ema_btc_ratio"]].round(3).to_string(index=False))

    # ===== 圖表 =====
    SHOW = {"純 DCA": "tab:orange",
            "DCA+CC 10% 常態": "tab:green",
            "DCA+CC 25% 常態": "tab:olive",
            "DCA+CC 10% EMA上": "tab:blue"}
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1])
    ax = fig.add_subplot(gs[0, :])
    for name, color in SHOW.items():
        df = store[name]
        ax.plot(df.index, df["value"], color=color, lw=1.2, label=name)
    cidx = pd.date_range(store["純 DCA"].index[0], periods=nm, freq="MS")
    ax.plot(cidx, np.arange(1, nm + 1) * CONTRIB, "k--", lw=1, label="累計投入")
    ax.set_title(f"主視窗 {MAIN_START}~：DCA + covered call（otm 3%, 1天期）法幣淨值")
    ax.set_ylabel("USDT"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    for name, color in SHOW.items():
        df = store[name]
        ax.plot(df.index, df["btc"], color=color, lw=1.1, label=name)
    ax.set_title("累積持幣量 (BTC)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    for name, color in SHOW.items():
        if name == "純 DCA":
            continue
        df = store[name]
        ax.plot(df.index, df["prem_cum"], color=color, lw=1.1, label=name)
    ax.set_title("累計權利金 (USDT)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "main_overview.png", dpi=130)

    print("\n輸出：results/exp10_dca_cc/")


if __name__ == "__main__":
    main()
