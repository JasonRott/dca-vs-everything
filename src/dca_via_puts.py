"""exp16：定期定額抄底寶 —— 月投進池、全池持續賣 1 天期價外 put 等接貨。

機制：
- 每月投入 contrib 進 USDT 池（吃 5% 息）。
- 每 UTC 日：先結算昨日 put（開盤 < 履約價 → 保證金以履約價換幣），
  再以全部未保留現金賣出 1 天期、價外 otm 的 cash-secured put，權利金入池。
- 幣只進不出（屯幣導向）。對照：純 DCA（月初市價買）。

問題：「等 3% 回檔再買 + 收權利金」能否勝過「定時即刻買」？
（exp11 已證無權利金版的「等回檔買」必輸；本實驗量化權利金能否翻盤）
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
from coin_accum import HOURLY_R
from dca_covered_call import run_dca_cc
from dual_wheel import bs_put
from fetch_binance import load_klines
from fetch_deribit import load_dvol
from grid_vs_dca import CONTRIB, FEE, iter_months, window_metrics

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False
RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp16_dca_via_puts"
RESULTS.mkdir(parents=True, exist_ok=True)

MAIN_START = "2023-07-01"
OTM_SWEEP = [0.02, 0.03, 0.05]


def run_dca_via_puts(bars: pd.DataFrame, iv: pd.Series, otm: float,
                     contrib: float = CONTRIB):
    pool = qty = 0.0
    live = None                  # (保證金, K)
    cur_day = None
    n_p = ex_p = 0
    prem_tot = spent = 0.0
    vals, btcs, idx = [], [], []

    for mstart, mbars in iter_months(bars):
        pool += contrib
        idx.extend(mbars.index)
        for ts, o, h, l, c in mbars[["open", "high", "low", "close"]] \
                .itertuples(index=True):
            pool *= (1 + HOURLY_R)
            op = float(o)
            day = ts.normalize()
            if day != cur_day:
                cur_day = day
                if live is not None:                 # 結算昨日 put
                    coll, kk = live
                    if op < kk:
                        pool -= coll
                        qty += coll / kk
                        spent += coll
                        ex_p += 1
                    live = None
                sigma = iv.asof(ts)
                if np.isfinite(sigma) and pool > 1.0:
                    coll = pool                      # 全池賣 put
                    kk = op * (1 - otm)
                    prem = bs_put(op, kk, float(sigma) / 100, 1 / 365) \
                        * (coll / kk)
                    pool += prem
                    prem_tot += prem
                    live = (coll, kk)
                    n_p += 1
            vals.append(pool + qty * c)
            btcs.append(qty)

    df = pd.DataFrame({"value": vals, "btc": btcs},
                      index=pd.DatetimeIndex(idx))
    diag = {"otm": otm, "final_btc": qty, "n_puts": n_p, "ex_puts": ex_p,
            "premium": prem_tot,
            "avg_cost": spent / qty if qty > 0 else np.nan,
            "undeployed": pool}
    return diag, df


def eval_window(sub, iv, label):
    nm = sum(1 for _ in iter_months(sub))
    _, ddf = run_dca_cc(sub, 0.0)
    dmet = window_metrics(ddf["value"], nm, CONTRIB)
    dqty = float(ddf["btc"].iloc[-1])
    dca_cost = nm * CONTRIB / dqty
    print(f"\n=== {label}（{nm} 個月）===")
    print(f"  純 DCA: MOIC {dmet['moic']:.3f}, IRR {dmet['irr_ann']:+.1%}, "
          f"MDD {dmet['max_dd']:.0%}, BTC {dqty:.4f}, 均價 {dca_cost:,.0f}")
    rows = []
    for otm in OTM_SWEEP:
        diag, df = run_dca_via_puts(sub, iv, otm)
        met = window_metrics(df["value"], nm, CONTRIB)
        rows.append({"window": label, **diag, **met,
                     "btc_ratio": diag["final_btc"] / dqty,
                     "cost_ratio": diag["avg_cost"] / dca_cost})
        print(f"  抄底寶DCA otm={otm:.0%}: MOIC {met['moic']:.3f}, "
              f"IRR {met['irr_ann']:+.1%}, MDD {met['max_dd']:.0%}, "
              f"BTC {diag['final_btc']:.4f} ({diag['final_btc']/dqty:.0%}), "
              f"均價比 {diag['avg_cost']/dca_cost:.3f}, "
              f"權利金 {diag['premium']:,.0f}, "
              f"履約 {diag['ex_puts']}/{diag['n_puts']}, "
              f"未部署 {diag['undeployed']:,.0f}")
    return rows


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")
    iv = load_dvol()["close"].shift(1)

    all_rows = []
    all_rows += eval_window(bars.loc[MAIN_START:], iv, f"主視窗 {MAIN_START}~")
    all_rows += eval_window(bars.loc["2022-12-01":"2024-12-01"], iv,
                            "牛市視窗 2022-12~2024-11")
    all_rows += eval_window(bars.loc["2021-12-01":"2023-01-01"], iv,
                            "熊市視窗 2021-12~2022-12")
    pd.DataFrame(all_rows).to_csv(RESULTS / "windows.csv", index=False)

    # 滾動 24 個月（otm=3%）
    starts = pd.date_range("2021-06-01", "2024-06-01", freq="3MS", tz="UTC")
    roll = []
    for st in starts:
        sub = bars.loc[st:st + pd.DateOffset(months=24)]
        nm = sum(1 for _ in iter_months(sub))
        if nm < 24:
            continue
        _, ddf = run_dca_cc(sub, 0.0)
        dmet = window_metrics(ddf["value"], nm, CONTRIB)
        dqty = float(ddf["btc"].iloc[-1])
        diag, df = run_dca_via_puts(sub, iv, 0.03)
        met = window_metrics(df["value"], nm, CONTRIB)
        roll.append({"start": st, "dca_moic": dmet["moic"],
                     "put_moic": met["moic"],
                     "dca_mdd": dmet["max_dd"], "put_mdd": met["max_dd"],
                     "btc_ratio": diag["final_btc"] / dqty})
    rdf = pd.DataFrame(roll)
    rdf.to_csv(RESULTS / "rolling24m.csv", index=False)
    print("\n=== 滾動 24 個月（otm=3%）===")
    print(rdf.assign(start=rdf["start"].dt.strftime("%Y-%m"))
          .round(3).to_string(index=False))
    wr = (rdf["put_moic"] > rdf["dca_moic"]).mean()
    bd = (rdf["put_mdd"] > rdf["dca_mdd"]).mean()
    print(f"\n  MOIC 勝率 vs DCA：{wr:.0%} | MDD 較淺比例：{bd:.0%} | "
          f"幣量比均值：{rdf['btc_ratio'].mean():.0%}")

    print("\n輸出：results/exp16_dca_via_puts/")


if __name__ == "__main__":
    main()
