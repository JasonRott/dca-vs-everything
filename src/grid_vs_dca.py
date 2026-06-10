"""上帝視角網格（含 ε 緩衝）vs 現貨 DCA — 多時間窗、等額月投入比較。

規則：
- 兩策略每月第一棒以相同金額 C 投入。
- 網格：上下限 = [月低×(1+ε), 月高×(1−ε)]，ε=1%；格數由 k×該月日波動率
  推出（同 god_view_backtest）；月底全清算，淨值 + C 開下月新網格。
- DCA：每月第一棒開盤市價買入 C 金額的 BTC，持有不賣。
- 手續費皆 0.1%。

指標：MOIC（期末淨值/總投入）、年化 IRR（月現金流）、最大回撤（逐時淨值）。
輸出：results/ 下的 CSV 與三張圖（全期路徑、分窗路徑、滾動視窗）。
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
from god_view_backtest import month_stats
from grid_engine import GeometricGridBot

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp02_god_vs_dca"
RESULTS.mkdir(parents=True, exist_ok=True)

FEE = 0.001
EPS = 0.01           # 上下限內縮緩衝
CONTRIB = 1_000.0    # 每月投入（USDT）
MAX_GRIDS = 500

WINDOWS = [
    ("全期間 18/01-26/05", "2018-01-01", "2026-06-01"),
    ("19/01-20/12 回升牛", "2019-01-01", "2021-01-01"),
    ("21/01-22/12 頂部進場", "2021-01-01", "2023-01-01"),
    ("22/01-22/12 純熊市", "2022-01-01", "2023-01-01"),
    ("23/01-24/12 復甦牛", "2023-01-01", "2025-01-01"),
    ("24/06-26/05 近兩年", "2024-06-01", "2026-06-01"),
]


def iter_months(bars: pd.DataFrame):
    for mstart, mbars in bars.groupby(pd.Grouper(freq="MS")):
        if len(mbars) >= 24 * 20:
            yield mstart, mbars


def run_grid_contrib(bars: pd.DataFrame, k: float, eps: float = EPS,
                     contrib: float = CONTRIB):
    """月投入制上帝視角網格。回傳 (逐月明細, 逐時淨值)。"""
    net = 0.0
    rows, curves = [], []
    for mstart, mbars in iter_months(bars):
        s = month_stats(mbars)
        lo, hi = s["low"] * (1 + eps), s["high"] * (1 - eps)
        capital = net + contrib
        spacing = k * s["vol_daily"]
        n = int(np.clip(round(np.log(hi / lo) / np.log1p(spacing)), 2, MAX_GRIDS))
        bot = GeometricGridBot(lo, hi, n, capital, fee=FEE)
        res = bot.run(mbars)
        rows.append({
            "month": mstart.strftime("%Y-%m"), "n_grids": n,
            "start_value": capital, "end_value": res.final_value,
            "grid_ret": res.final_value / capital - 1,
            "btc_ret": s["close"] / s["open"] - 1,
            "fees": res.fees_paid, "avg_exposure": float(res.exposure.mean()),
        })
        curves.append(res.value_curve)
        net = res.final_value
    return pd.DataFrame(rows), pd.concat(curves)


def run_dca(bars: pd.DataFrame, contrib: float = CONTRIB):
    """月定額買入持有。回傳 (逐月明細, 逐時淨值)。"""
    qty = 0.0
    rows, curves = [], []
    for mstart, mbars in iter_months(bars):
        p0 = float(mbars["open"].iloc[0])
        qty += contrib * (1 - FEE) / p0
        rows.append({"month": mstart.strftime("%Y-%m"), "buy_price": p0,
                     "qty_cum": qty})
        curves.append(mbars["close"] * qty)
    return pd.DataFrame(rows), pd.concat(curves)


def irr_annual(n_months: int, contrib: float, final: float) -> float:
    """月現金流 IRR（雙分法），年化。"""
    t = np.arange(n_months)

    def npv(r):
        return float(-contrib * ((1 + r) ** -t).sum() + final * (1 + r) ** -n_months)

    lo_r, hi_r = -0.95, 10.0
    if npv(lo_r) * npv(hi_r) > 0:
        return np.nan
    for _ in range(200):
        mid = (lo_r + hi_r) / 2
        if npv(lo_r) * npv(mid) <= 0:
            hi_r = mid
        else:
            lo_r = mid
    return (1 + mid) ** 12 - 1


def max_dd(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min())


def window_metrics(curve: pd.Series, n_months: int, contrib: float) -> dict:
    final = float(curve.iloc[-1])
    invested = contrib * n_months
    return {"final": final, "invested": invested, "moic": final / invested,
            "irr_ann": irr_annual(n_months, contrib, final),
            "max_dd": max_dd(curve)}


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")

    # ---- (0) ε 緩衝下的 k 敏感度（全期間，月投入制） ----
    print("=== k 敏感度（ε=1%, 全期間, 月投入 1000）===")
    sweep = []
    for k in [0.5, 1.0, 2.0, 3.0, 6.0]:
        m, c = run_grid_contrib(bars, k)
        met = window_metrics(c, len(m), CONTRIB)
        sweep.append({"k": k, "avg_n": m["n_grids"].mean(), **met})
        print(f"  k={k}: MOIC {met['moic']:.2f}, IRR {met['irr_ann']:+.1%}, "
              f"MDD {met['max_dd']:.1%}, 平均格數 {m['n_grids'].mean():.1f}")
    pd.DataFrame(sweep).to_csv(RESULTS / "eps_k_sweep.csv", index=False)
    k_best = max(sweep, key=lambda r: r["moic"])["k"]
    K_SHOW = 2.0   # 圖表用代表值（穩健中段）；k_best 另列於表

    # ---- (1) 各時間窗比較 ----
    print(f"\n=== 時間窗比較（網格 k={K_SHOW} 與 k={k_best}; DCA）===")
    win_rows, win_curves = [], {}
    for name, ws, we in WINDOWS:
        sub = bars.loc[ws:we]
        gm, gc = run_grid_contrib(sub, K_SHOW)
        gm2, gc2 = run_grid_contrib(sub, k_best)
        dm, dc = run_dca(sub)
        n = len(gm)
        g, g2, d = (window_metrics(gc, n, CONTRIB),
                    window_metrics(gc2, n, CONTRIB),
                    window_metrics(dc, n, CONTRIB))
        win_rows.append({
            "window": name, "months": n,
            "grid_moic": g["moic"], "grid_irr": g["irr_ann"], "grid_mdd": g["max_dd"],
            "gridB_moic": g2["moic"], "gridB_irr": g2["irr_ann"], "gridB_mdd": g2["max_dd"],
            "dca_moic": d["moic"], "dca_irr": d["irr_ann"], "dca_mdd": d["max_dd"],
        })
        win_curves[name] = (gc, dc, n)
        print(f"  {name}: 網格 MOIC {g['moic']:.2f} / DCA {d['moic']:.2f} | "
              f"MDD {g['max_dd']:.0%} vs {d['max_dd']:.0%}")
    wdf = pd.DataFrame(win_rows)
    wdf.to_csv(RESULTS / "windows_grid_vs_dca.csv", index=False)

    # ---- (2) 滾動 24 個月視窗 ----
    starts = pd.date_range("2018-01-01", "2024-06-01", freq="3MS", tz="UTC")
    roll = []
    for st in starts:
        en = st + pd.DateOffset(months=24)
        sub = bars.loc[st:en]
        gm, gc = run_grid_contrib(sub, K_SHOW)
        if len(gm) < 24:
            continue
        dm, dc = run_dca(sub)
        g = window_metrics(gc, len(gm), CONTRIB)
        d = window_metrics(dc, len(gm), CONTRIB)
        roll.append({"start": st, "grid_moic": g["moic"], "dca_moic": d["moic"],
                     "grid_mdd": g["max_dd"], "dca_mdd": d["max_dd"]})
    rdf = pd.DataFrame(roll)
    rdf.to_csv(RESULTS / "rolling24m_grid_vs_dca.csv", index=False)

    # ================= 圖表 =================
    # 圖 1：全期路徑
    name0 = WINDOWS[0][0]
    gc, dc, n0 = win_curves[name0]
    contrib_line = pd.Series(
        [CONTRIB * (i // 1 + 1) for i in range(n0)],
        index=pd.date_range(gc.index[0], periods=n0, freq="MS"))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(gc.index, gc.values, label=f"上帝視角網格 (k={K_SHOW}, ε=1%)",
            color="tab:blue")
    ax.plot(dc.index, dc.values, label="現貨 DCA", color="tab:orange", alpha=0.85)
    ax.plot(contrib_line.index, contrib_line.values, "k--", lw=1.2,
            label="累計投入本金")
    ax.set_yscale("log")
    ax.set_title(f"月投入 {CONTRIB:.0f} USDT：上帝視角網格 vs 現貨 DCA（2018-2026，對數刻度）")
    ax.set_ylabel("淨值 (USDT)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "vs_dca_full_path.png", dpi=130)

    # 圖 2：各時間窗路徑（線性刻度）
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (name, _, _) in zip(axes.flat, WINDOWS):
        gc, dc, n = win_curves[name]
        cl = np.arange(1, n + 1) * CONTRIB
        cidx = pd.date_range(gc.index[0], periods=n, freq="MS")
        ax.plot(gc.index, gc.values, color="tab:blue", label="網格")
        ax.plot(dc.index, dc.values, color="tab:orange", alpha=0.85, label="DCA")
        ax.plot(cidx, cl, "k--", lw=1, label="投入本金")
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle("各時間窗：上帝視角網格(ε=1%) vs DCA 淨值路徑", fontsize=13)
    fig.tight_layout(); fig.savefig(RESULTS / "vs_dca_windows.png", dpi=130)

    # 圖 3：滾動 24 個月視窗
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = rdf["start"]
    axes[0].plot(x, rdf["grid_moic"], "-o", ms=4, label="網格 MOIC", color="tab:blue")
    axes[0].plot(x, rdf["dca_moic"], "-o", ms=4, label="DCA MOIC", color="tab:orange")
    axes[0].axhline(1, color="gray", lw=0.8)
    axes[0].set_title("滾動 24 個月視窗：期末淨值/總投入 (MOIC)")
    axes[0].set_xlabel("視窗起始月"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(x, rdf["grid_mdd"], "-o", ms=4, label="網格 MDD", color="tab:blue")
    axes[1].plot(x, rdf["dca_mdd"], "-o", ms=4, label="DCA MDD", color="tab:orange")
    axes[1].set_title("滾動 24 個月視窗：最大回撤")
    axes[1].set_xlabel("視窗起始月"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "vs_dca_rolling24m.png", dpi=130)

    print("\n輸出：results/eps_k_sweep.csv, windows_grid_vs_dca.csv, "
          "rolling24m_grid_vs_dca.csv 與三張 PNG")


if __name__ == "__main__":
    main()
