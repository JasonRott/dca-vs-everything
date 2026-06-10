"""exp11：純掛買單階梯（限價單版 DCA）——「網格買法」的最純粹檢驗。

機制：每月預算（月投 1000 + 變體b的滾存）拆成 5 張等額限價買單，
掛在月初開盤價下方 [D/5, 2D/5, ..., D]（D = 階梯深度）。只買不賣。
- 變體 a「月底補成交」：月底未成交的預算以收盤市價補買——與 DCA 等量投入，
  純粹比較「等回檔買 vs 即刻買」的均價差異。
- 變體 b「滾存等待」：未成交預算滾入下月階梯（吃不到息：掛單資金鎖定）。
  承擔「漲勢中永遠買不到」的部署風險。

實務上這不需要機器人：就是 5 張現貨限價單，每月初重掛一次。
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
from fetch_binance import load_klines
from grid_vs_dca import CONTRIB, FEE, iter_months, run_dca, window_metrics

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp11_buy_ladder"
RESULTS.mkdir(parents=True, exist_ok=True)

MAIN_START = "2023-07-01"
N_LEVELS = 5
DEPTH_SWEEP = [0.05, 0.10, 0.15]


def run_ladder(bars: pd.DataFrame, depth: float, mode: str,
               contrib: float = CONTRIB, n_levels: int = N_LEVELS):
    """mode: "fill_eom"（月底補成交）或 "rollover"（滾存等待）。"""
    pool = 0.0          # 月間未投入階梯的資金（本設計中月初全數投入）
    qty = spent = 0.0
    placed = filled = 0
    vals, btcs, idx = [], [], []

    for mstart, mbars in iter_months(bars):
        pool += contrib
        p0 = float(mbars["open"].iloc[0])
        per = pool / n_levels
        levels = [p0 * (1 - depth * i / n_levels) for i in range(1, n_levels + 1)]
        open_orders = {lv: per for lv in levels}
        pool = 0.0
        placed += n_levels
        idx.extend(mbars.index)
        for ts, o, h, l, c in mbars[["open", "high", "low", "close"]] \
                .itertuples(index=True):
            for lv in [x for x in open_orders if l <= x]:
                amt = open_orders.pop(lv)
                qty += amt * (1 - FEE) / lv
                spent += amt
                filled += 1
            cash_left = sum(open_orders.values())
            vals.append(qty * c + cash_left)
            btcs.append(qty)
        # 月底處理未成交
        rem = sum(open_orders.values())
        if rem > 0:
            if mode == "fill_eom":
                close = float(mbars["close"].iloc[-1])
                qty += rem * (1 - FEE) / close
                spent += rem
                vals[-1] = qty * close
                btcs[-1] = qty
            else:
                pool = rem
    df = pd.DataFrame({"value": vals, "btc": btcs},
                      index=pd.DatetimeIndex(idx))
    diag = {"depth": depth, "mode": mode, "final_btc": qty,
            "avg_cost": spent / qty if qty > 0 else np.nan,
            "spent": spent, "undeployed": pool,
            "fill_rate": filled / max(placed, 1)}
    return diag, df


def eval_window(sub: pd.DataFrame) -> pd.DataFrame:
    nm = sum(1 for _ in iter_months(sub))
    dm, dc = run_dca(sub)
    dqty = float(dm["qty_cum"].iloc[-1])
    dmet = window_metrics(dc, nm, CONTRIB)
    rows = [{"strategy": "DCA", "moic": dmet["moic"], "irr_ann": dmet["irr_ann"],
             "max_dd": dmet["max_dd"], "btc": dqty, "btc_ratio": 1.0,
             "avg_cost": nm * CONTRIB / dqty, "cost_ratio": 1.0,
             "fill_rate": 1.0, "undeployed": 0.0}]
    for depth in DEPTH_SWEEP:
        for mode, mname in [("fill_eom", "月底補成交"), ("rollover", "滾存等待")]:
            diag, df = run_ladder(sub, depth, mode)
            met = window_metrics(df["value"], nm, CONTRIB)
            rows.append({"strategy": f"階梯 D={depth:.0%} {mname}",
                         "moic": met["moic"], "irr_ann": met["irr_ann"],
                         "max_dd": met["max_dd"], "btc": diag["final_btc"],
                         "btc_ratio": diag["final_btc"] / dqty,
                         "avg_cost": diag["avg_cost"],
                         "cost_ratio": diag["avg_cost"] / (nm * CONTRIB / dqty),
                         "fill_rate": diag["fill_rate"],
                         "undeployed": diag["undeployed"]})
    return pd.DataFrame(rows)


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")

    # ===== 主視窗 =====
    sub = bars.loc[MAIN_START:]
    mdf = eval_window(sub)
    mdf.to_csv(RESULTS / "main_window.csv", index=False)
    print(f"=== 主視窗 {MAIN_START}~ ===")
    print(mdf[["strategy", "moic", "irr_ann", "max_dd", "btc_ratio",
               "cost_ratio", "fill_rate", "undeployed"]]
          .round(3).to_string(index=False))

    # ===== 滾動 24 個月（全歷史）=====
    starts = pd.date_range("2018-01-01", "2024-06-01", freq="3MS", tz="UTC")
    roll = []
    for st in starts:
        s2 = bars.loc[st:st + pd.DateOffset(months=24)]
        if sum(1 for _ in iter_months(s2)) < 24:
            continue
        w = eval_window(s2).set_index("strategy")
        r = {"start": st}
        for tag, key in [("dca", "DCA"),
                         ("eom10", "階梯 D=10% 月底補成交"),
                         ("roll10", "階梯 D=10% 滾存等待")]:
            r[f"{tag}_moic"] = w.loc[key, "moic"]
            r[f"{tag}_btc_ratio"] = w.loc[key, "btc_ratio"]
            r[f"{tag}_cost_ratio"] = w.loc[key, "cost_ratio"]
        roll.append(r)
    rdf = pd.DataFrame(roll)
    rdf.to_csv(RESULTS / "rolling24m.csv", index=False)
    yr = rdf.assign(yr=rdf["start"].dt.year).groupby("yr")[
        ["dca_moic", "eom10_moic", "roll10_moic",
         "eom10_btc_ratio", "roll10_btc_ratio",
         "eom10_cost_ratio", "roll10_cost_ratio"]].mean().round(3)
    print("\n=== 滾動視窗依起始年平均（D=10%）===")
    print(yr.to_string())
    yr.to_csv(RESULTS / "rolling_by_year.csv")

    # ===== 圖表 =====
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    ax.plot(rdf["start"], rdf["dca_moic"], "-o", ms=3, color="tab:orange",
            label="DCA")
    ax.plot(rdf["start"], rdf["eom10_moic"], "-o", ms=3, color="tab:green",
            label="階梯10% 月底補")
    ax.plot(rdf["start"], rdf["roll10_moic"], "-o", ms=3, color="tab:blue",
            label="階梯10% 滾存")
    ax.axhline(1, color="gray", lw=0.8)
    ax.set_title("滾動 24 個月 MOIC"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(rdf["start"], rdf["eom10_cost_ratio"], "-o", ms=3,
            color="tab:green", label="階梯10% 月底補")
    ax.plot(rdf["start"], rdf["roll10_cost_ratio"], "-o", ms=3,
            color="tab:blue", label="階梯10% 滾存")
    ax.axhline(1, color="tab:orange", lw=1.2, label="DCA 均價 = 1")
    ax.set_title("取得均價 / DCA（<1 = 更便宜）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot(rdf["start"], rdf["eom10_btc_ratio"], "-o", ms=3,
            color="tab:green", label="階梯10% 月底補")
    ax.plot(rdf["start"], rdf["roll10_btc_ratio"], "-o", ms=3,
            color="tab:blue", label="階梯10% 滾存")
    ax.axhline(1, color="tab:orange", lw=1.2, label="DCA = 1")
    ax.set_title("期末持幣量 / DCA"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for axx in axes:
        axx.set_xlabel("視窗起始月")
    fig.suptitle("exp11：純掛買單階梯 vs DCA（滾動 24 個月）", fontsize=13)
    fig.tight_layout(); fig.savefig(RESULTS / "rolling.png", dpi=130)

    print("\n輸出：results/exp11_buy_ladder/")


if __name__ == "__main__":
    main()
