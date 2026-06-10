"""資訊階梯比較：上帝視角 vs 歷史資訊 vs 完全無資訊 vs DCA。

三個網格變體共用規則（月投入 1000、月底全清算滾入下月、ε 觀念僅上帝視角適用、
出界後等到月底才重置、手續費 0.1%）：

1. 上帝視角（上一輪的 run_grid_contrib，k=2, ε=1%）— 資訊上限。
2. 歷史資訊版：僅用「該月開盤前」的歷史日波動率（lookback 180 天）。
   - 加權：simple = sqrt(mean(r^2))；ewma = 半衰期 30 天的指數加權。
   - 範圍：上下限 = 開盤價 × exp(±c·σ_d·√30.4)（GBM 分位數映射），c sweep。
   - 間距：k·σ_d，k sweep。格數 n ≈ 2c√30.4/k，與 σ 無關。
3. 完全無資訊版：上下限 = 開盤價 × (1±20%)，等比間距 s 直接 sweep。

前視偏差控制：波動率估計只用 t < 月初 的日線資料（2017-08 起的日線）。
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
from grid_vs_dca import (CONTRIB, FEE, MAX_GRIDS, WINDOWS, iter_months,
                         run_dca, run_grid_contrib, window_metrics)

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp03_info_ladder"
RESULTS.mkdir(parents=True, exist_ok=True)
LOOKBACK_D = 180
HALF_LIFE_D = 30
DAYS_PER_MONTH = 30.44
GOD_K = 2.0

C_SWEEP = [1.0, 1.5, 2.0]
K_SWEEP = [0.5, 1.0, 2.0, 3.0, 6.0]
S_SWEEP = [0.01, 0.02, 0.03, 0.05, 0.10]   # 無資訊版等比間距


# ---------- 波動率估計（嚴格使用月初之前的資料） ----------
def build_sigma_maps(daily: pd.DataFrame) -> dict[str, dict]:
    ret = np.log(daily["close"]).diff().dropna()
    months = pd.date_range("2018-01-01", "2026-06-01", freq="MS", tz="UTC")
    simple, ewma = {}, {}
    for ms in months:
        w = ret[(ret.index < ms) & (ret.index >= ms - pd.Timedelta(days=LOOKBACK_D))]
        if len(w) < 60:
            continue
        r2 = w.values ** 2
        simple[ms] = float(np.sqrt(r2.mean()))
        age = (ms - w.index).days.values.astype(float)
        wts = 0.5 ** (age / HALF_LIFE_D)
        ewma[ms] = float(np.sqrt((wts * r2).sum() / wts.sum()))
    return {"simple": simple, "ewma": ewma}


# ---------- 歷史資訊版 ----------
def run_hist(bars: pd.DataFrame, sigma_map: dict, c: float, k: float,
             contrib: float = CONTRIB):
    net = 0.0
    rows, curves = [], []
    for mstart, mbars in iter_months(bars):
        sigma = sigma_map.get(mstart)
        if sigma is None:
            continue
        p0 = float(mbars["open"].iloc[0])
        half_width = c * sigma * np.sqrt(DAYS_PER_MONTH)
        lo, hi = p0 * np.exp(-half_width), p0 * np.exp(half_width)
        n = int(np.clip(round(np.log(hi / lo) / np.log1p(k * sigma)), 2, MAX_GRIDS))
        capital = net + contrib
        bot = GeometricGridBot(lo, hi, n, capital, fee=FEE)
        res = bot.run(mbars)
        breach = "down" if mbars["low"].min() < lo else (
                 "up" if mbars["high"].max() > hi else "in")
        rows.append({"month": mstart.strftime("%Y-%m"), "sigma": sigma,
                     "n_grids": n, "breach": breach,
                     "start_value": capital, "end_value": res.final_value,
                     "grid_ret": res.final_value / capital - 1,
                     "avg_exposure": float(res.exposure.mean())})
        curves.append(res.value_curve)
        net = res.final_value
    return pd.DataFrame(rows), pd.concat(curves)


# ---------- 完全無資訊版 ----------
def run_noinfo(bars: pd.DataFrame, spacing: float, width: float = 0.20,
               contrib: float = CONTRIB):
    net = 0.0
    rows, curves = [], []
    for mstart, mbars in iter_months(bars):
        p0 = float(mbars["open"].iloc[0])
        lo, hi = p0 * (1 - width), p0 * (1 + width)
        n = int(np.clip(round(np.log(hi / lo) / np.log1p(spacing)), 2, MAX_GRIDS))
        capital = net + contrib
        bot = GeometricGridBot(lo, hi, n, capital, fee=FEE)
        res = bot.run(mbars)
        rows.append({"month": mstart.strftime("%Y-%m"), "n_grids": n,
                     "start_value": capital, "end_value": res.final_value,
                     "grid_ret": res.final_value / capital - 1,
                     "avg_exposure": float(res.exposure.mean())})
        curves.append(res.value_curve)
        net = res.final_value
    return pd.DataFrame(rows), pd.concat(curves)


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")
    daily = load_klines("BTCUSDT", "1d", "2017-08-17", "2026-06-01")
    sig = build_sigma_maps(daily)

    # ===== (1) 歷史資訊版 sweep =====
    print("=== 歷史資訊版 sweep（全期間）===")
    hrows = []
    for wname, smap in sig.items():
        for c in C_SWEEP:
            for k in K_SWEEP:
                m, cv = run_hist(bars, smap, c, k)
                met = window_metrics(cv, len(m), CONTRIB)
                hrows.append({"weight": wname, "c": c, "k": k,
                              "avg_n": m["n_grids"].mean(),
                              "breach_dn": (m["breach"] == "down").mean(),
                              "breach_up": (m["breach"] == "up").mean(),
                              **met})
    hdf = pd.DataFrame(hrows).sort_values("moic", ascending=False)
    hdf.to_csv(RESULTS / "hist_sweep.csv", index=False)
    print(hdf.head(8)[["weight", "c", "k", "avg_n", "moic", "irr_ann",
                       "max_dd", "breach_dn", "breach_up"]].to_string(index=False))
    best_h = hdf.iloc[0]
    print(f"→ 最佳：weight={best_h['weight']}, c={best_h['c']}, k={best_h['k']}")

    # ===== (2) 無資訊版 sweep =====
    print("\n=== 無資訊版 sweep（±20%, 全期間）===")
    nrows = []
    for s in S_SWEEP:
        m, cv = run_noinfo(bars, s)
        met = window_metrics(cv, len(m), CONTRIB)
        nrows.append({"spacing": s, "avg_n": m["n_grids"].mean(), **met})
        print(f"  s={s:.0%}: MOIC {met['moic']:.2f}, IRR {met['irr_ann']:+.1%}, "
              f"MDD {met['max_dd']:.1%}, n={m['n_grids'].mean():.0f}")
    ndf = pd.DataFrame(nrows)
    ndf.to_csv(RESULTS / "noinfo_sweep.csv", index=False)
    best_n = ndf.loc[ndf["moic"].idxmax()]
    print(f"→ 最佳：spacing={best_n['spacing']:.0%}")

    # ===== (3) 四策略：時間窗 + 滾動視窗 =====
    smap = sig[best_h["weight"]]
    cH, kH, sN = float(best_h["c"]), float(best_h["k"]), float(best_n["spacing"])

    def run_all(sub: pd.DataFrame) -> dict[str, tuple]:
        out = {}
        gm, gc = run_grid_contrib(sub, GOD_K)
        out["上帝視角"] = (gm, gc)
        hm, hc = run_hist(sub, smap, cH, kH)
        out["歷史資訊"] = (hm, hc)
        nm, nc = run_noinfo(sub, sN)
        out["無資訊±20%"] = (nm, nc)
        dm, dc = run_dca(sub)
        out["DCA"] = (dm, dc)
        return out

    print("\n=== 時間窗比較（MOIC | MDD）===")
    win_rows, win_store = [], {}
    for name, ws, we in WINDOWS:
        res = run_all(bars.loc[ws:we])
        n = len(res["上帝視角"][0])
        row = {"window": name, "months": n}
        line = f"  {name}:"
        for sname, (m, cv) in res.items():
            met = window_metrics(cv, len(m), CONTRIB)
            key = {"上帝視角": "god", "歷史資訊": "hist",
                   "無資訊±20%": "noinfo", "DCA": "dca"}[sname]
            row[f"{key}_moic"], row[f"{key}_irr"], row[f"{key}_mdd"] = \
                met["moic"], met["irr_ann"], met["max_dd"]
            line += f" {sname} {met['moic']:.2f}|{met['max_dd']:.0%} "
        win_rows.append(row)
        win_store[name] = res
        print(line)
    pd.DataFrame(win_rows).to_csv(RESULTS / "ladder_windows.csv", index=False)

    starts = pd.date_range("2018-01-01", "2024-06-01", freq="3MS", tz="UTC")
    roll = []
    for st in starts:
        sub = bars.loc[st:st + pd.DateOffset(months=24)]
        res = run_all(sub)
        if len(res["上帝視角"][0]) < 24:
            continue
        r = {"start": st}
        for sname, (m, cv) in res.items():
            key = {"上帝視角": "god", "歷史資訊": "hist",
                   "無資訊±20%": "noinfo", "DCA": "dca"}[sname]
            met = window_metrics(cv, len(m), CONTRIB)
            r[f"{key}_moic"], r[f"{key}_mdd"] = met["moic"], met["max_dd"]
        roll.append(r)
    rdf = pd.DataFrame(roll)
    rdf.to_csv(RESULTS / "ladder_rolling24m.csv", index=False)

    # ===== 圖表 =====
    COLORS = {"上帝視角": "tab:green", "歷史資訊": "tab:blue",
              "無資訊±20%": "tab:red", "DCA": "tab:orange"}
    full = win_store[WINDOWS[0][0]]

    fig, ax = plt.subplots(figsize=(12, 6))
    for sname, (m, cv) in full.items():
        ax.plot(cv.index, cv.values, label=sname, color=COLORS[sname],
                alpha=0.9, lw=1.3)
    n0 = len(full["DCA"][0])
    cidx = pd.date_range(full["DCA"][1].index[0], periods=n0, freq="MS")
    ax.plot(cidx, np.arange(1, n0 + 1) * CONTRIB, "k--", lw=1.1, label="累計投入本金")
    ax.set_yscale("log")
    ax.set_title(f"資訊階梯：月投入 {CONTRIB:.0f} USDT 的四策略淨值路徑（2018-2026，對數刻度）\n"
                 f"歷史資訊版: {best_h['weight']}, c={cH}, k={kH} | 無資訊版: ±20%, 間距 {sN:.0%}")
    ax.set_ylabel("淨值 (USDT)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "ladder_full_path.png", dpi=130)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for axx, (name, _, _) in zip(axes.flat, WINDOWS):
        res = win_store[name]
        for sname, (m, cv) in res.items():
            axx.plot(cv.index, cv.values, color=COLORS[sname], alpha=0.9,
                     lw=1.1, label=sname)
        n = len(res["DCA"][0])
        ci = pd.date_range(res["DCA"][1].index[0], periods=n, freq="MS")
        axx.plot(ci, np.arange(1, n + 1) * CONTRIB, "k--", lw=0.9, label="投入本金")
        axx.set_title(name, fontsize=11); axx.grid(alpha=0.3)
        axx.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("各時間窗：資訊階梯四策略淨值路徑", fontsize=13)
    fig.tight_layout(); fig.savefig(RESULTS / "ladder_windows.png", dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for sname, key in [("上帝視角", "god"), ("歷史資訊", "hist"),
                       ("無資訊±20%", "noinfo"), ("DCA", "dca")]:
        axes[0].plot(rdf["start"], rdf[f"{key}_moic"], "-o", ms=3,
                     color=COLORS[sname], label=sname)
        axes[1].plot(rdf["start"], rdf[f"{key}_mdd"], "-o", ms=3,
                     color=COLORS[sname], label=sname)
    axes[0].axhline(1, color="gray", lw=0.8)
    axes[0].set_yscale("log")
    axes[0].set_title("滾動 24 個月視窗 MOIC（對數刻度）")
    axes[1].set_title("滾動 24 個月視窗最大回撤")
    for axx in axes:
        axx.set_xlabel("視窗起始月"); axx.legend(fontsize=9); axx.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "ladder_rolling24m.png", dpi=130)

    print("\n輸出：results/hist_sweep.csv, noinfo_sweep.csv, ladder_windows.csv, "
          "ladder_rolling24m.csv 與三張 ladder_*.png")


if __name__ == "__main__":
    main()
