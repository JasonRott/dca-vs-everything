"""exp04：DGT 機制等價版重現，與全部既有策略比較。

DGT 規則（依官方程式碼 grid_logic.py 的機制，做三處本研究調整）：
- 以當前價 P 為中心，上下各 m 格（共 n=2m），啟動時上半部以市價買入半倉。
- 上穿上限：全數變現回收（此時持倉已沿途賣空），利潤滾入錢包，以新價為中心重開。
- 下穿下限：全部持幣轉入「永不賣出的 COIN 桶」（不實現虧損），以新價為中心重開。
- 本研究調整：(1) 等比網格（步長 g，levels = P(1+g)^i, i=-m..m），與既有實驗可比；
  (2) 1h 資料 + 0.1% 手續費（密格成交數為保守下限）；
  (3) 錢包制資金流：月投 1000 進錢包，每次（重）開網格本金 = 錢包全額，
      錢包為零（下穿後必然如此）則暫停至下次月投。與其他策略現金流等額。

對照：上帝視角(k=2,ε=1%)、歷史資訊(ewma,c=2,k=6)、無資訊(±20%,間距10%)、DCA。
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
from grid_engine import GeometricGridBot
from grid_vs_dca import (CONTRIB, FEE, WINDOWS, iter_months, run_dca,
                         run_grid_contrib, window_metrics)
from realistic_vs_dca import build_sigma_maps, run_hist, run_noinfo

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp04_dgt"
RESULTS.mkdir(parents=True, exist_ok=True)

GOD_K = 2.0
HIST_CFG = dict(c=2.0, k=6.0)      # exp03 最佳（ewma 加權）
NOINFO_S = 0.10

G_SWEEP = [0.01, 0.02, 0.03, 0.05]
M_SWEEP = [2, 3, 5, 7, 10]


def run_dgt(bars: pd.DataFrame, g: float, m: int, contrib: float = CONTRIB):
    """機制等價 DGT。回傳 (診斷 dict, 逐時淨值)。"""
    n = 2 * m
    wallet, coin = 0.0, 0.0
    bot: GeometricGridBot | None = None
    resets_up = resets_dn = 0
    paused_bars = 0
    vals, idx = [], []  # idx 逐月累積，避免不完整月份造成索引錯位

    def open_grid(p: float):
        nonlocal wallet, bot
        if wallet <= 0:
            bot = None
            return
        lo, hi = p * (1 + g) ** -m, p * (1 + g) ** m
        bot = GeometricGridBot(lo, hi, n, wallet, fee=FEE)
        bot.start(p)
        wallet = 0.0

    for mstart, mbars in iter_months(bars):
        wallet += contrib
        idx.extend(mbars.index)
        for o, h, l, c in mbars[["open", "high", "low", "close"]] \
                .itertuples(index=False):
            if bot is None:
                open_grid(float(o))
            if bot is None:
                paused_bars += 1
            else:
                path = (o, l, h, c) if c >= o else (o, h, l, c)
                for p in path:
                    p = float(p)
                    if bot is None:
                        break
                    bot.move_to(p)
                    if p > bot.levels[-1]:          # 上穿：全現金回收重開
                        wallet += bot.cash + bot.total_qty * p * (1 - FEE)
                        resets_up += 1
                        open_grid(p)
                    elif p < bot.levels[0]:         # 下穿：持幣入桶、重開
                        coin += bot.total_qty
                        wallet += bot.cash
                        resets_dn += 1
                        open_grid(p)
            grid_val = (bot.cash + bot.total_qty * c) if bot is not None else 0.0
            vals.append(wallet + grid_val + coin * c)
    curve = pd.Series(vals, index=pd.DatetimeIndex(idx))
    last_close = float(bars["close"].loc[curve.index[-1]])
    diag = {"g": g, "m": m, "resets_up": resets_up, "resets_dn": resets_dn,
            "paused_frac": paused_bars / max(len(vals), 1),
            "coin_qty": coin, "coin_value": coin * last_close,
            "coin_share": coin * last_close / vals[-1] if vals[-1] > 0 else 0}
    return diag, curve


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")
    daily = load_klines("BTCUSDT", "1d", "2017-08-17", "2026-06-01")
    smap = build_sigma_maps(daily)["ewma"]

    # ===== (1) DGT 參數掃描（全期間）=====
    print("=== DGT sweep（全期間, 月投 1000）===")
    rows = []
    for g in G_SWEEP:
        for m in M_SWEEP:
            diag, cv = run_dgt(bars, g, m)
            n_months = 101
            met = window_metrics(cv, n_months, CONTRIB)
            rows.append({**diag, **met})
            print(f"  g={g:.0%}, m={m} (半寬{(1+g)**m-1:+.0%}): MOIC {met['moic']:.2f}, "
                  f"IRR {met['irr_ann']:+.1%}, MDD {met['max_dd']:.1%}, "
                  f"重置 ↑{diag['resets_up']}/↓{diag['resets_dn']}, "
                  f"幣桶占比 {diag['coin_share']:.0%}")
    sdf = pd.DataFrame(rows).sort_values("moic", ascending=False)
    sdf.to_csv(RESULTS / "dgt_sweep.csv", index=False)
    best = sdf.iloc[0]
    gB, mB = float(best["g"]), int(best["m"])
    print(f"→ 最佳：g={gB:.0%}, m={mB}")

    # ===== (2) 五策略比較 =====
    def run_all(sub: pd.DataFrame) -> dict:
        out = {}
        out["上帝視角"] = run_grid_contrib(sub, GOD_K)[1]
        out["歷史資訊"] = run_hist(sub, smap, HIST_CFG["c"], HIST_CFG["k"])[1]
        out["無資訊±20%"] = run_noinfo(sub, NOINFO_S)[1]
        out["DGT"] = run_dgt(sub, gB, mB)[1]
        out["DCA"] = run_dca(sub)[1]
        return out

    KEY = {"上帝視角": "god", "歷史資訊": "hist", "無資訊±20%": "noinfo",
           "DGT": "dgt", "DCA": "dca"}
    COLORS = {"上帝視角": "tab:green", "歷史資訊": "tab:blue",
              "無資訊±20%": "tab:red", "DGT": "tab:purple", "DCA": "tab:orange"}

    print("\n=== 時間窗比較（MOIC | MDD）===")
    win_rows, win_store = [], {}
    for name, ws, we in WINDOWS:
        sub = bars.loc[ws:we]
        curves = run_all(sub)
        n_months = sum(1 for _ in iter_months(sub))
        row, line = {"window": name, "months": n_months}, f"  {name}:"
        for sname, cv in curves.items():
            met = window_metrics(cv, n_months, CONTRIB)
            row[f"{KEY[sname]}_moic"] = met["moic"]
            row[f"{KEY[sname]}_irr"] = met["irr_ann"]
            row[f"{KEY[sname]}_mdd"] = met["max_dd"]
            line += f" {sname} {met['moic']:.2f}|{met['max_dd']:.0%} "
        win_rows.append(row); win_store[name] = curves
        print(line)
    pd.DataFrame(win_rows).to_csv(RESULTS / "dgt_windows.csv", index=False)

    starts = pd.date_range("2018-01-01", "2024-06-01", freq="3MS", tz="UTC")
    roll = []
    for st in starts:
        sub = bars.loc[st:st + pd.DateOffset(months=24)]
        n_months = sum(1 for _ in iter_months(sub))
        if n_months < 24:
            continue
        curves = run_all(sub)
        r = {"start": st}
        for sname, cv in curves.items():
            met = window_metrics(cv, n_months, CONTRIB)
            r[f"{KEY[sname]}_moic"], r[f"{KEY[sname]}_mdd"] = met["moic"], met["max_dd"]
        roll.append(r)
    rdf = pd.DataFrame(roll)
    rdf.to_csv(RESULTS / "dgt_rolling24m.csv", index=False)

    # ===== 圖表 =====
    full = win_store[WINDOWS[0][0]]
    fig, ax = plt.subplots(figsize=(12, 6))
    for sname, cv in full.items():
        ax.plot(cv.index, cv.values, label=sname, color=COLORS[sname],
                lw=1.2, alpha=0.9)
    n0 = 101
    cidx = pd.date_range(full["DCA"].index[0], periods=n0, freq="MS")
    ax.plot(cidx, np.arange(1, n0 + 1) * CONTRIB, "k--", lw=1, label="累計投入本金")
    ax.set_yscale("log")
    ax.set_title(f"五策略淨值路徑（月投 1000 USDT，2018-2026，對數刻度）\n"
                 f"DGT: g={gB:.0%}, m={mB}（機制等價重現, 錢包制）")
    ax.set_ylabel("淨值 (USDT)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "dgt_full_path.png", dpi=130)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for axx, (name, _, _) in zip(axes.flat, WINDOWS):
        curves = win_store[name]
        for sname, cv in curves.items():
            axx.plot(cv.index, cv.values, color=COLORS[sname], lw=1.0,
                     alpha=0.9, label=sname)
        nm = len(curves["DCA"].resample("MS").first())
        ci = pd.date_range(curves["DCA"].index[0], periods=nm, freq="MS")
        axx.plot(ci, np.arange(1, nm + 1) * CONTRIB, "k--", lw=0.9)
        axx.set_title(name, fontsize=11); axx.grid(alpha=0.3)
        axx.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("各時間窗：五策略淨值路徑", fontsize=13)
    fig.tight_layout(); fig.savefig(RESULTS / "dgt_windows.png", dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for sname, key in KEY.items():
        axes[0].plot(rdf["start"], rdf[f"{key}_moic"], "-o", ms=3,
                     color=COLORS[sname], label=sname)
        axes[1].plot(rdf["start"], rdf[f"{key}_mdd"], "-o", ms=3,
                     color=COLORS[sname], label=sname)
    axes[0].axhline(1, color="gray", lw=0.8); axes[0].set_yscale("log")
    axes[0].set_title("滾動 24 個月視窗 MOIC（對數刻度）")
    axes[1].set_title("滾動 24 個月視窗最大回撤")
    for axx in axes:
        axx.set_xlabel("視窗起始月"); axx.legend(fontsize=9); axx.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "dgt_rolling24m.png", dpi=130)

    print("\n輸出：results/exp04_dgt/")


if __name__ == "__main__":
    main()
