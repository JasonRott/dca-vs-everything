"""exp12：DGT 完全仿造驗證（完整檢驗）。

引擎：逐行移植官方 repo（colachenkc/Dynamic-Grid-Trading）的封閉公式記帳——
等差網格（步長 = 重置中心價 × grid_size）、上破回收本金+公式利潤、
下破半倉+下方全部買入轉入永不賣出的 COIN、無限母池（fund_next_grid：
錢包不足即外部注資補滿固定本金）。參數採官方 config：本金 100、
fee 0.0008、BTC 1m 資料 2021-01-01 ~ 2024-07-31。

評估指標改用本研究標準：
- 含時間戳現金流的 XIRR（取代論文把所有注資當期初投入的 (V/I)^(12/43)）
- 現金流配對買入持有（同時點、同金額注資直接買 BTC 持有）作為對照
- MTM 淨值曲線（網格段以帳面本金近似，誤差 ≤ 本金×網格半寬，相對 COIN
  桶可忽略）與最大回撤

官方程式碼既有問題（移植時修正/保留註記）：
1. dgt_backtest.py 對 handle_down_break / settle_last_grid_segment 的呼叫
   多傳 max_price 參數，與函式簽名不符 → 發布版無法執行（TypeError）。
   本移植移除該參數。
2. 棒內路徑固定為 開→低→高→收，不分陰陽棒（系統性先觸下後觸上）。保留。
3. calculate_profit_down 從未被呼叫（死碼）。
4. IRR 公式 (V/I)^(12/43) 寫死 43 個月且把分批注資視為期初投入。
   本實驗兩種都算，以對照記帳法的影響。
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

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp12_dgt_faithful"
RESULTS.mkdir(parents=True, exist_ok=True)

PRINCIPAL = 100.0
FEE = 0.0008
SIZES = [0.005, 0.01, 0.02, 0.05, 0.10]
MS = [2, 5, 10]


# ---------- 官方公式（逐行移植） ----------
def grid_levels_of(start_price, grid_size, m):
    delta = start_price * grid_size
    lv = [start_price - delta * i for i in range(m, 0, -1)]
    lv.append(start_price)
    lv += [start_price + delta * i for i in range(1, m + 1)]
    return lv


def profit_up(n, m, size):
    return (m * (m + 1) / 2) * (PRINCIPAL / n) * (size - FEE * 2)


def profit_arb(n, m, size, tc):
    return ((tc - m) / 2) * (PRINCIPAL / n) * (size - FEE * 2)


# ---------- 主回測（官方 dgt_backtest.py 邏輯；期末結算內聯於迴圈後） ----------
def run_dgt_paper(bars: pd.DataFrame, grid_size: float, m: int):
    n = 2 * m
    USDT = COIN = 0.0
    money_input = PRINCIPAL
    t0 = bars.index[0]
    p_first = float(bars["open"].iloc[0])
    flows = [(t0, PRINCIPAL, p_first)]          # (時間, 注資金額, 當時價格)
    init_price = p_first
    levels = grid_levels_of(init_price, grid_size, m)
    lower, upper = levels[0], levels[-1]
    cl = m
    tc = 0
    n_up = n_dn = 0
    mtm_t, mtm_v = [], []

    def fund_next(ts, price):
        nonlocal USDT, money_input
        if USDT >= PRINCIPAL:
            USDT -= PRINCIPAL
        else:
            need = PRINCIPAL - USDT
            money_input += need
            flows.append((ts, need, price))
            USDT = 0.0

    arr = bars[["open", "low", "high", "close"]].to_numpy()
    idx_arr = bars.index
    prev = None
    for i in range(len(arr)):
        o, lo, hi, c = arr[i]
        prices = (prev if prev is not None else o, lo, hi, c)
        ts = idx_arr[i]
        for k in range(3):
            a, b = prices[k], prices[k + 1]
            if a < b:
                while cl < n and a <= levels[cl + 1] < b:
                    cl += 1
                    tc += 1
            else:
                while cl > 0 and b <= levels[cl - 1] < a:
                    cl -= 1
                    tc += 1
            if b > upper or cl == n:               # 上破
                USDT += profit_up(n, m, grid_size) \
                    + profit_arb(n, m, grid_size, tc) + PRINCIPAL
                n_up += 1
                tc = 0
                fund_next(ts, b)
                init_price = b
                levels = grid_levels_of(init_price, grid_size, m)
                lower, upper = levels[0], levels[-1]
                cl = m
            if b < lower or cl == 0:               # 下破
                USDT += profit_arb(n, m, grid_size, tc)
                COIN += (PRINCIPAL / 2) / init_price * (1 - FEE * 2)
                for j in range(m):
                    COIN += (PRINCIPAL / n) / levels[j] * (1 - FEE * 2)
                n_dn += 1
                tc = 0
                fund_next(ts, b)
                init_price = b
                levels = grid_levels_of(init_price, grid_size, m)
                lower, upper = levels[0], levels[-1]
                cl = m
        prev = c
        if i % 60 == 0:                            # 每小時取樣 MTM
            mtm_t.append(ts)
            mtm_v.append(USDT + COIN * c + PRINCIPAL)

    # 期末結算（官方 settle 的等價簡化：剩餘網格以期末收盤清算）
    close_end = float(arr[-1, 3])
    from bisect import bisect_right
    per = PRINCIPAL / n
    mid_coin = (PRINCIPAL / 2) / levels[m] * (1 - FEE)
    pos = bisect_right(levels, close_end) - 1
    if close_end >= levels[m]:
        remain = n - pos
        up_count = max(0, pos - m)
        USDT += remain * per
        USDT += up_count * per + profit_up(n, up_count, grid_size) \
            + profit_arb(n, up_count, grid_size, tc)
        COIN += mid_coin * (remain / m) if m > 0 else 0.0
    else:
        down_count = max(0, m - pos)
        for j in range(m - 1, max(pos, -1), -1):
            COIN += (per / levels[j]) * (1 - FEE * 2)
        USDT += profit_arb(n, down_count, grid_size, tc)
        USDT += (m - down_count) * per
        COIN += mid_coin

    final = USDT + COIN * close_end
    mtm_t.append(idx_arr[-1])
    mtm_v.append(final)
    curve = pd.Series(mtm_v, index=pd.DatetimeIndex(mtm_t))
    return {"size": grid_size, "m": m, "final": final,
            "money_input": money_input, "usdt": USDT, "coin": COIN,
            "coin_value": COIN * close_end,
            "n_up": n_up, "n_dn": n_dn,
            "flows": flows, "curve": curve, "close_end": close_end}


# ---------- 指標 ----------
def xirr(flows_dated: list[tuple], final_ts, final_val) -> float:
    """flows_dated: [(ts, 投入金額>0, _)]；終值為正流入。"""
    t0 = flows_dated[0][0]
    cfs = [(-(amt), (ts - t0).total_seconds() / 86400 / 365.25)
           for ts, amt, _ in flows_dated]
    cfs.append((final_val, (final_ts - t0).total_seconds() / 86400 / 365.25))

    def npv(r):
        return sum(a / (1 + r) ** t for a, t in cfs)

    lo_r, hi_r = -0.9999, 100.0
    if npv(lo_r) * npv(hi_r) > 0:
        return np.nan
    for _ in range(200):
        mid = (lo_r + hi_r) / 2
        if npv(lo_r) * npv(mid) <= 0:
            hi_r = mid
        else:
            lo_r = mid
    return mid


def flow_matched_bh(flows, close_end, final_ts):
    qty = sum(amt * (1 - FEE) / px for _, amt, px in flows)
    final = qty * close_end
    return final, xirr(flows, final_ts, final)


def main():
    bars = load_klines("BTCUSDT", "1m", "2021-01-01", "2024-08-01")
    print(f"資料：{len(bars):,} 根 1m，{bars.index[0]} ~ {bars.index[-1]}")
    months = (bars.index[-1] - bars.index[0]).days / 30.44

    rows = []
    best = None
    for size in SIZES:
        for m in MS:
            r = run_dgt_paper(bars, size, m)
            total_in = r["money_input"]
            moic = r["final"] / total_in
            paper_irr = moic ** (12 / months) - 1
            ir = xirr(r["flows"], bars.index[-1], r["final"])
            bh_final, bh_irr = flow_matched_bh(r["flows"], r["close_end"],
                                               bars.index[-1])
            dd = float((r["curve"] / r["curve"].cummax() - 1).min())
            rows.append({"size": size, "m": m, "total_input": total_in,
                         "final": r["final"], "moic": moic,
                         "paper_irr": paper_irr, "xirr": ir,
                         "bh_final": bh_final, "bh_moic": bh_final / total_in,
                         "bh_xirr": bh_irr, "max_dd": dd,
                         "coin_share": r["coin_value"] / r["final"],
                         "n_up": r["n_up"], "n_dn": r["n_dn"],
                         "n_inject": len(r["flows"])})
            print(f"size={size:.3f}, m={m}: 注資 {total_in:,.0f} ({len(r['flows'])}筆), "
                  f"終值 {r['final']:,.0f}, MOIC {moic:.3f}, "
                  f"論文式IRR {paper_irr:+.1%}, XIRR {ir:+.1%} | "
                  f"配對B&H: MOIC {bh_final/total_in:.3f}, XIRR {bh_irr:+.1%} | "
                  f"MDD {dd:.0%}, 重置 ↑{r['n_up']}/↓{r['n_dn']}")
            if best is None or moic > best[0]:
                best = (moic, size, m, r)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "dgt_faithful_sweep.csv", index=False)

    # 最佳組合圖：MTM vs 配對 B&H、注資事件
    _, sizeB, mB, rB = best
    flows = rB["flows"]
    qty = 0.0
    bh_t, bh_v = [], []
    fi = 0
    closes = bars["close"].iloc[::60]
    for ts, px in closes.items():
        while fi < len(flows) and flows[fi][0] <= ts:
            qty += flows[fi][1] * (1 - FEE) / flows[fi][2]
            fi += 1
        bh_t.append(ts)
        bh_v.append(qty * px)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(rB["curve"].index, rB["curve"].values, color="tab:purple",
                 lw=1.1, label=f"DGT 忠實版 (size={sizeB:.1%}, m={mB})")
    axes[0].plot(bh_t, bh_v, color="tab:orange", lw=1.1, alpha=0.85,
                 label="現金流配對買入持有")
    cum_in = np.cumsum([f[1] for f in flows])
    axes[0].step([f[0] for f in flows], cum_in, where="post", color="k",
                 ls="--", lw=1, label="累計注資")
    axes[0].set_yscale("log"); axes[0].set_ylabel("USDT")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[0].set_title("DGT 忠實重現（官方引擎+無限母池）vs 現金流配對 B&H")
    inj_t = [f[0] for f in flows[1:]]
    axes[1].hist(inj_t, bins=60, color="tab:red", alpha=0.7)
    axes[1].set_title("外部注資事件分布（下破救援）")
    axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "dgt_faithful.png", dpi=130)
    print("\n輸出：results/exp12_dgt_faithful/")


if __name__ == "__main__":
    main()
