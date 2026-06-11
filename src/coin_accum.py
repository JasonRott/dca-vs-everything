"""exp05:屯幣版網格（第一部分：不含選擇權、不看指標）。

規則：
- 每月投入 CONTRIB 至 USDT 池；池內閒置資金以 5% APY 逐時複利（期現套利收益）。
- 開網格：從池中取 ratio 比例，±15% 上下限、固定等比間距 s、標準半倉啟動。
- 上破上限：網格內 USDT 全數回池，立即以新價開新網格。
- 下破下限：網格內 BTC 全數轉入「永存桶」（記錄成本基礎），殘餘現金回池，
  立即以新價開新網格（池×ratio 不足 1 USDT 則暫停，資金到位後下一棒開）。
- 雙重記帳：法幣（MOIC/IRR/MDD）+ 屯幣（期末 BTC、取得均價）+ 曝險路徑。
"""
from __future__ import annotations

import math
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
from grid_vs_dca import CONTRIB, FEE, WINDOWS, iter_months, run_dca, window_metrics

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp05_coin_accum"
RESULTS.mkdir(parents=True, exist_ok=True)

APY = 0.05
HOURLY_R = APY / (365 * 24)
WIDTH = 0.15
RATIO_SWEEP = [0.25, 0.50, 0.75, 1.00]
S_SWEEP = [0.01, 0.02, 0.03, 0.05]


def bs_call(s: float, k: float, sigma: float, t: float) -> float:
    """Black-Scholes 歐式買權（r=0）。sigma 年化、t 年。"""
    if sigma <= 0 or t <= 0 or s <= 0:
        return max(s - k, 0.0)
    v = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * v * v) / v
    d2 = d1 - v
    nd = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return s * nd(d1) - k * nd(d2)


def run_accum(bars: pd.DataFrame, ratio: float, spacing: float,
              contrib: float = CONTRIB,
              ema: pd.Series | None = None, ema_mode: str | None = None,
              down_reopen: str = "immediate",
              call_frac: float = 0.0, call_otm: float = 0.03,
              call_trigger: str = "always",
              iv: pd.Series | None = None, iv_mult: float = 1.0,
              width: float = WIDTH):
    """屯幣版網格。回傳 (診斷 dict, DataFrame[value, exposure, btc])。

    ema_mode（搭配 ema：與 bars 同索引、無前視的 200 日 EMA）：
      - "no_open_below"：開網格時價格 < EMA → 不建初始底倉（純掛買單）
      - "no_open_above"：開網格時價格 > EMA → 不建初始底倉
      - None：永遠建標準半倉
    down_reopen：
      - "immediate"：下破後立即以新價重開（exp05/06 行為）
      - "next_contrib"：下破後進入冷卻，等到下一次單期投入才允許重開
      - "ema_conditional"：下破時價格 < 200EMA 才冷卻（陰跌防接刀），
        價格 ≥ EMA 則立即重開（牛市回檔吃 V 反彈）；需傳入 ema

    covered call 層（call_frac > 0 且傳入 iv 時啟用；止盈寶機制）：
      - 每個 UTC 日第一棒：先結算昨日批次（開盤價 > 履約價 → 該批幣以履約價
        賣出、款項入池；否則保留），再賣出新批次 = call_frac × 幣桶，
        履約價 = 現價 × (1+call_otm)，1 天到期。
      - 權利金 = Black-Scholes(現價, K, DVOL前日收盤×iv_mult, 1/365)，USDT 入池。
      - call_trigger："always" 常態輪動；"above_ema" 僅價格 > 200EMA 時賣出。
      - iv：日線 DVOL 收盤（%、年化），呼叫端需先 shift(1) 防前視。
    """
    pool = 0.0
    bucket_qty = bucket_cost = 0.0
    bot: GeometricGridBot | None = None
    resets_up = resets_dn = 0
    interest_earned = 0.0
    opens_with_pos = opens_without_pos = 0
    paused_bars = 0
    can_open = True
    cur_day = None
    call_live: tuple[float, float] | None = None   # (數量, 履約價)
    n_calls = n_exercised = 0
    premium_total = called_qty = called_proceeds = 0.0
    vals, expo, btcs, prems, idx = [], [], [], [], []

    def open_grid(p: float, ts) -> None:
        nonlocal pool, bot, opens_with_pos, opens_without_pos
        amt = pool * ratio
        if amt < 1.0:
            bot = None
            return
        init_pos = True
        if ema is not None and ema_mode is not None:
            e = float(ema.loc[ts])
            if np.isfinite(e):
                if ema_mode == "no_open_below" and p < e:
                    init_pos = False
                elif ema_mode == "no_open_above" and p > e:
                    init_pos = False
        lo, hi = p * (1 - width), p * (1 + width)
        n = max(2, int(round(np.log(hi / lo) / np.log1p(spacing))))
        bot = GeometricGridBot(lo, hi, n, amt, fee=FEE)
        bot.start(p, initial_position=init_pos)
        if init_pos:
            opens_with_pos += 1
        else:
            opens_without_pos += 1
        pool -= amt

    def to_bucket(price_hint: float):
        """下破：網格持幣連同成本基礎轉入永存桶。"""
        nonlocal bucket_qty, bucket_cost, pool, bot
        bucket_qty += bot.total_qty
        bucket_cost += float(bot.cost[bot.has_coin].sum())
        pool += bot.cash
        bot = None

    for mstart, mbars in iter_months(bars):
        pool += contrib
        can_open = True          # 單期投入到位，解除下破冷卻
        idx.extend(mbars.index)
        for ts, o, h, l, c in mbars[["open", "high", "low", "close"]] \
                .itertuples(index=True):
            gain = pool * HOURLY_R
            pool += gain
            interest_earned += gain
            # ---- covered call 日輪動（結算昨日批次 → 賣出新批次）----
            day = ts.normalize()
            if call_frac > 0 and iv is not None and day != cur_day:
                cur_day = day
                op = float(o)
                if call_live is not None:
                    qc, kk = call_live
                    if op > kk and bucket_qty > 0:      # 履約：以 K 賣出
                        share = min(qc / bucket_qty, 1.0)
                        bucket_cost -= bucket_cost * share
                        bucket_qty -= qc
                        pool += qc * kk
                        called_qty += qc
                        called_proceeds += qc * kk
                        n_exercised += 1
                    call_live = None
                sell_ok = bucket_qty > 0
                if call_trigger == "above_ema" and ema is not None:
                    e = float(ema.loc[ts])
                    sell_ok = sell_ok and np.isfinite(e) and op > e
                if sell_ok:
                    sigma = iv.asof(ts)
                    if np.isfinite(sigma):
                        qc = call_frac * bucket_qty
                        kk = op * (1 + call_otm)
                        prem = bs_call(op, kk, float(sigma) * iv_mult / 100,
                                       1 / 365) * qc
                        pool += prem
                        premium_total += prem
                        call_live = (qc, kk)
                        n_calls += 1
            if bot is None and can_open:
                open_grid(float(o), ts)
            if bot is None:
                paused_bars += 1
            if bot is not None:
                path = (o, l, h, c) if c >= o else (o, h, l, c)
                for p in path:
                    p = float(p)
                    if bot is None:
                        break
                    bot.move_to(p)
                    if p > bot.levels[-1]:        # 上破：USDT 回池重開
                        pool += bot.cash + bot.total_qty * p * (1 - FEE)
                        resets_up += 1
                        open_grid(p, ts)
                    elif p < bot.levels[0]:       # 下破：BTC 入桶
                        to_bucket(p)
                        resets_dn += 1
                        if down_reopen == "next_contrib":
                            can_open = False      # 冷卻至下次單期投入
                        elif down_reopen == "ema_conditional":
                            e = float(ema.loc[ts]) if ema is not None else np.nan
                            if np.isfinite(e) and p < e:
                                can_open = False  # EMA 下的下破才冷卻
                            else:
                                open_grid(p, ts)
                        else:
                            open_grid(p, ts)
            grid_cash = bot.cash if bot is not None else 0.0
            grid_qty = bot.total_qty if bot is not None else 0.0
            total_btc = bucket_qty + grid_qty
            v = pool + grid_cash + total_btc * c
            vals.append(v)
            expo.append(total_btc * c / v if v > 0 else 0.0)
            btcs.append(total_btc)
            prems.append(premium_total)

    df = pd.DataFrame({"value": vals, "exposure": expo, "btc": btcs,
                       "prem_cum": prems},
                      index=pd.DatetimeIndex(idx))
    diag = {
        "ratio": ratio, "spacing": spacing,
        "opens_with_pos": opens_with_pos, "opens_without_pos": opens_without_pos,
        "resets_up": resets_up, "resets_dn": resets_dn,
        "paused_frac": paused_bars / max(len(vals), 1),
        "n_calls": n_calls, "n_exercised": n_exercised,
        "premium_total": premium_total,
        "called_qty": called_qty, "called_proceeds": called_proceeds,
        "final_btc": btcs[-1], "bucket_btc": bucket_qty,
        "bucket_avg_cost": bucket_cost / bucket_qty if bucket_qty > 0 else np.nan,
        "final_pool": pool, "interest": interest_earned,
        "avg_exposure": float(np.mean(expo)),
    }
    return diag, df


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")
    n_months_full = sum(1 for _ in iter_months(bars))

    # ---- DCA 基準（含屯幣指標） ----
    dca_m, dca_curve = run_dca(bars)
    dca_qty = dca_m["qty_cum"].iloc[-1]
    dca_avg_cost = (n_months_full * CONTRIB) / dca_qty
    dca_met = window_metrics(dca_curve, n_months_full, CONTRIB)

    # ---- sweep ----
    print("=== exp05 sweep（全期間 101 個月，月投 1000）===")
    rows, store = [], {}
    for ratio in RATIO_SWEEP:
        for s in S_SWEEP:
            diag, df = run_accum(bars, ratio, s)
            met = window_metrics(df["value"], n_months_full, CONTRIB)
            rows.append({**diag, **met,
                         "btc_vs_dca": diag["final_btc"] / dca_qty,
                         "cost_vs_dca": diag["bucket_avg_cost"] / dca_avg_cost})
            store[(ratio, s)] = df
            print(f"  ratio={ratio:.0%}, s={s:.0%}: MOIC {met['moic']:.2f}, "
                  f"MDD {met['max_dd']:.0%}, BTC {diag['final_btc']:.3f} "
                  f"({diag['final_btc']/dca_qty:.0%} of DCA), "
                  f"屯幣均價 {diag['bucket_avg_cost']:,.0f}, "
                  f"平均曝險 {diag['avg_exposure']:.0%}, "
                  f"重置 ↑{diag['resets_up']}/↓{diag['resets_dn']}")
    sdf = pd.DataFrame(rows)
    sdf.to_csv(RESULTS / "accum_sweep.csv", index=False)
    print(f"\nDCA 基準：MOIC {dca_met['moic']:.2f}, MDD {dca_met['max_dd']:.0%}, "
          f"BTC {dca_qty:.3f}, 均價 {dca_avg_cost:,.0f}, 曝險 100%")

    best = sdf.loc[sdf["moic"].idxmax()]
    rB, sB = float(best["ratio"]), float(best["spacing"])

    # ---- 時間窗（最佳組合 vs DCA，雙重記帳）----
    print(f"\n=== 時間窗（ratio={rB:.0%}, s={sB:.0%} vs DCA）===")
    wrows = []
    for name, ws, we in WINDOWS:
        sub = bars.loc[ws:we]
        nm = sum(1 for _ in iter_months(sub))
        diag, df = run_accum(sub, rB, sB)
        met = window_metrics(df["value"], nm, CONTRIB)
        dm, dc = run_dca(sub)
        dqty = dm["qty_cum"].iloc[-1]
        dmet = window_metrics(dc, nm, CONTRIB)
        wrows.append({"window": name,
                      "moic": met["moic"], "mdd": met["max_dd"],
                      "btc": diag["final_btc"], "btc_vs_dca": diag["final_btc"] / dqty,
                      "avg_expo": diag["avg_exposure"],
                      "dca_moic": dmet["moic"], "dca_mdd": dmet["max_dd"],
                      "dca_btc": dqty})
        print(f"  {name}: MOIC {met['moic']:.2f}|{met['max_dd']:.0%} "
              f"BTC {diag['final_btc']/dqty:.0%} of DCA, 曝險 {diag['avg_exposure']:.0%} "
              f"|| DCA {dmet['moic']:.2f}|{dmet['max_dd']:.0%}")
    pd.DataFrame(wrows).to_csv(RESULTS / "accum_windows.csv", index=False)

    # ===== 圖表（代表組：間距固定 3%，ratio 掃描）=====
    show = [(r, 0.03) for r in RATIO_SWEEP]
    cmap = {0.25: "tab:blue", 0.50: "tab:green", 0.75: "tab:purple", 1.00: "tab:red"}

    # 圖1：法幣淨值
    fig, ax = plt.subplots(figsize=(12, 6))
    for r, s in show:
        df = store[(r, s)]
        ax.plot(df.index, df["value"], color=cmap[r], lw=1.1,
                label=f"屯幣網格 ratio={r:.0%}")
    ax.plot(dca_curve.index, dca_curve.values, color="tab:orange", lw=1.2,
            alpha=0.85, label="DCA")
    cidx = pd.date_range(dca_curve.index[0], periods=n_months_full, freq="MS")
    ax.plot(cidx, np.arange(1, n_months_full + 1) * CONTRIB, "k--", lw=1,
            label="累計投入本金")
    ax.set_yscale("log")
    ax.set_title("屯幣版網格（±15%, 間距3%, 閒錢5%APY）vs DCA：法幣淨值（對數刻度）")
    ax.set_ylabel("淨值 (USDT)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "accum_value.png", dpi=130)

    # 圖2：曝險走勢（本實驗核心圖）
    fig, axes = plt.subplots(len(show) + 1, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(bars.index, bars["close"], color="gray", lw=0.8)
    axes[0].set_yscale("log"); axes[0].set_ylabel("BTC 價格")
    axes[0].set_title("曝險走勢：BTC 市值占總淨值比例（間距 3%）")
    axes[0].grid(alpha=0.3)
    for axx, (r, s) in zip(axes[1:], show):
        df = store[(r, s)]
        axx.fill_between(df.index, df["exposure"], color=cmap[r], alpha=0.55)
        axx.axhline(1.0, color="gray", lw=0.5)
        axx.set_ylim(0, 1.05); axx.set_ylabel(f"ratio={r:.0%}")
        axx.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "accum_exposure.png", dpi=130)

    # 圖3：屯幣量
    fig, ax = plt.subplots(figsize=(12, 5))
    for r, s in show:
        df = store[(r, s)]
        ax.plot(df.index, df["btc"], color=cmap[r], lw=1.1,
                label=f"ratio={r:.0%}")
    dq = dca_curve / bars["close"].reindex(dca_curve.index)   # DCA 持幣量
    ax.plot(dq.index, dq.values, color="tab:orange", lw=1.2, alpha=0.85,
            label="DCA")
    ax.set_title("累積持幣量（BTC，含網格內存貨與永存桶）")
    ax.set_ylabel("BTC"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "accum_btc.png", dpi=130)

    print("\n輸出：results/exp05_coin_accum/")


if __name__ == "__main__":
    main()
