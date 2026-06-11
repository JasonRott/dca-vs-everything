"""exp14：EMA 方向 + 固定比例雙輪（止盈寶/抄底寶的完整版原始構想）。

狀態機（200 日 EMA ± buffer 遲滯帶，防鋸齒）：
- 價格 > EMA×(1+buffer) → bull 狀態；< EMA×(1−buffer) → bear 狀態；
  之間維持原狀態。
- bull：月投市價買入（DCA 照常）；每日賣 1 天期 otm 3% call，
  規模 = frac × 持幣（固定比例 = exp10 已驗證的收入引擎）。
- bear：月投進現金池（不市價買，等抄底寶接貨）；每日賣 1 天期 otm 3% put，
  規模 = frac × 未保留現金；**曝險 > put_cap (80%) 時停賣**（防止熊市
  曝險單調上升的保險絲）。

測量重點（對應使用者三個疑慮）：
- 日曝險跳變分布（p95 / 最大值）——履約造成的瞬間曝險變化
- 熊市視窗（2021-12~2022-12，全年低於 EMA）的曝險軌跡——單調上升檢驗
- 與 exp13 帶控制器的防禦（MDD、曝險std）對比
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
from dual_wheel import bs_put, run_dual_wheel
from ema_accum import build_ema
from fetch_binance import load_klines
from fetch_deribit import load_dvol
from grid_vs_dca import CONTRIB, FEE, iter_months, window_metrics

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp14_ema_wheel"
RESULTS.mkdir(parents=True, exist_ok=True)

MAIN_START = "2023-07-01"
BEAR_WIN = ("2021-12-01", "2023-01-01")
OTM = 0.03
BUFFER = 0.03
PUT_CAP = 0.80
FRACS = [0.10, 0.25]


def run_ema_wheel(bars: pd.DataFrame, ema: pd.Series, iv: pd.Series,
                  frac: float, otm: float = OTM, buffer: float = BUFFER,
                  put_cap: float = PUT_CAP, contrib: float = CONTRIB,
                  put_frac: float | None = None, put_otm: float | None = None,
                  hysteresis: str = "price", confirm_days: int = 7):
    """hysteresis："price" = EMA×(1±buffer) 突破即切換（exp14 原版）；
    "time" = 連續 confirm_days 個交易日開盤在 EMA 另一側才切換，
    等待確認期間維持原 regime 行為（照常）。
    put_frac / put_otm：抄底寶側獨立參數（預設同 call 側）。"""
    put_frac = frac if put_frac is None else put_frac
    put_otm = otm if put_otm is None else put_otm
    pool = qty = reserved = 0.0
    live_call = live_put = None
    state = None                       # "bull" / "bear"
    opp_streak = 0                     # time 遲滯：連續反向日數
    cur_day = None
    n_c = n_p = ex_c = ex_p = 0
    prem_c = prem_p = 0.0
    vals, expos, btcs, prems, states, idx = [], [], [], [], [], []

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
                    if state is None:
                        state = "bull" if op >= e else "bear"
                    elif hysteresis == "time":
                        opposite = (op < e) if state == "bull" else (op > e)
                        opp_streak = opp_streak + 1 if opposite else 0
                        if opp_streak >= confirm_days:   # 一週未回頭→換框架
                            state = "bear" if state == "bull" else "bull"
                            opp_streak = 0
                    else:                                # price 遲滯（exp14）
                        if op > e * (1 + buffer):
                            state = "bull"
                        elif op < e * (1 - buffer):
                            state = "bear"
                if live_call is not None:
                    qc, kk = live_call
                    if op > kk and qty > 0:
                        qc = min(qc, qty)
                        qty -= qc
                        pool += qc * kk
                        ex_c += 1
                    live_call = None
                if live_put is not None:
                    coll, kk = live_put
                    reserved -= coll
                    if op < kk:
                        pool -= coll
                        qty += coll / kk
                        ex_p += 1
                    live_put = None
            if month_buy and state == "bull":          # bull 才市價月投
                buy = min(contrib, pool - reserved)
                if buy > 1e-9:
                    qty += buy * (1 - FEE) / op
                    pool -= buy
                month_buy = False
            elif month_buy and state == "bear":        # bear 進池等抄底寶
                month_buy = False
            if roll and state is not None:
                tot = pool + qty * op
                expo = qty * op / tot if tot > 0 else 0.0
                sigma = iv.asof(ts)
                if np.isfinite(sigma) and tot > 0:
                    sig = float(sigma) / 100
                    if state == "bull" and qty > 0:
                        qc = frac * qty
                        kk = op * (1 + otm)
                        prem = bs_call(op, kk, sig, 1 / 365) * qc
                        pool += prem
                        prem_c += prem
                        live_call = (qc, kk)
                        n_c += 1
                    elif state == "bear" and expo < put_cap \
                            and (pool - reserved) > 1.0:
                        coll = put_frac * (pool - reserved)
                        kk = op * (1 - put_otm)
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
            states.append(1 if state == "bull" else 0)

    df = pd.DataFrame({"value": vals, "exposure": expos, "btc": btcs,
                       "prem_cum": prems, "bull": states},
                      index=pd.DatetimeIndex(idx))
    e = df["exposure"]
    djump = e.resample("1D").last().diff().abs().dropna()
    diag = {"frac": frac,
            "final_btc": qty, "expo_mean": float(e.mean()),
            "expo_std": float(e.std()),
            "jump_p95": float(djump.quantile(0.95)),
            "jump_max": float(djump.max()),
            "bear_time": float(1 - df["bull"].mean()),
            "n_calls": n_c, "n_puts": n_p, "ex_calls": ex_c, "ex_puts": ex_p,
            "prem_call": prem_c, "prem_put": prem_p}
    return diag, df


def report_window(bars, ema, iv, label):
    nm = sum(1 for _ in iter_months(bars))
    _, dca_df = run_dca_cc(bars, 0.0)
    dca_met = window_metrics(dca_df["value"], nm, CONTRIB)
    dqty = float(dca_df["btc"].iloc[-1])
    print(f"\n=== {label}（{nm} 個月）===")
    print(f"  純 DCA: MOIC {dca_met['moic']:.3f}, IRR {dca_met['irr_ann']:+.1%}, "
          f"MDD {dca_met['max_dd']:.0%}, BTC {dqty:.4f}")
    diag13, df13 = run_dual_wheel(bars, (0.50, 0.70), iv, buy_target="dca")
    met13 = window_metrics(df13["value"], nm, CONTRIB)
    print(f"  exp13帶控[50,70]: MOIC {met13['moic']:.3f}, "
          f"IRR {met13['irr_ann']:+.1%}, MDD {met13['max_dd']:.0%}, "
          f"曝險 {diag13['expo_mean']:.0%}±{diag13['expo_std']:.0%}")
    rows, store = [], {}
    for frac in FRACS:
        diag, df = run_ema_wheel(bars, ema, iv, frac)
        met = window_metrics(df["value"], nm, CONTRIB)
        store[frac] = df
        rows.append({**diag, **met, "btc_ratio": diag["final_btc"] / dqty,
                     "window": label})
        print(f"  EMA雙輪 frac={frac:.0%}: MOIC {met['moic']:.3f}, "
              f"IRR {met['irr_ann']:+.1%}, MDD {met['max_dd']:.0%}, "
              f"BTC {diag['final_btc']:.4f} ({diag['final_btc']/dqty:.0%}), "
              f"曝險 {diag['expo_mean']:.0%}±{diag['expo_std']:.0%}, "
              f"日跳變 p95 {diag['jump_p95']:.1%}/max {diag['jump_max']:.1%}, "
              f"熊態時間 {diag['bear_time']:.0%}, "
              f"call {diag['n_calls']}/{diag['ex_calls']}, "
              f"put {diag['n_puts']}/{diag['ex_puts']}, "
              f"權利金 C{diag['prem_call']:,.0f}+P{diag['prem_put']:,.0f}")
    return rows, store, dca_df


def main():
    bars = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")
    daily = load_klines("BTCUSDT", "1d", "2017-08-17", "2026-06-01")
    ema = build_ema(daily, bars.index)
    iv = load_dvol()["close"].shift(1)

    all_rows = []
    r1, store_main, dca_main = report_window(
        bars.loc[MAIN_START:], ema, iv, f"主視窗 {MAIN_START}~")
    all_rows += r1
    r2, store_bear, _ = report_window(
        bars.loc[BEAR_WIN[0]:BEAR_WIN[1]], ema, iv,
        "熊市壓力測試 2021-12~2022-12")
    all_rows += r2
    pd.DataFrame(all_rows).to_csv(RESULTS / "windows.csv", index=False)

    # ===== 圖：主視窗總覽 + 熊市曝險軌跡 =====
    sub = bars.loc[MAIN_START:]
    nm = sum(1 for _ in iter_months(sub))
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1, 1])
    ax = fig.add_subplot(gs[0])
    ax.plot(dca_main.index, dca_main["value"], color="tab:orange", lw=1.3,
            label="純 DCA")
    for frac, color in [(0.10, "tab:blue"), (0.25, "tab:green")]:
        df = store_main[frac]
        ax.plot(df.index, df["value"], color=color, lw=1.0,
                label=f"EMA雙輪 frac={frac:.0%}")
    cidx = pd.date_range(dca_main.index[0], periods=nm, freq="MS")
    ax.plot(cidx, np.arange(1, nm + 1) * CONTRIB, "k--", lw=1, label="累計投入")
    ax.set_title(f"exp14 EMA 方向雙輪（{MAIN_START}~）：法幣淨值")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    for i, (frac, color) in enumerate([(0.10, "tab:blue"), (0.25, "tab:green")]):
        axx = fig.add_subplot(gs[1 + i])
        df = store_main[frac]
        axx.fill_between(df.index, 0, 1, where=df["bull"] == 0,
                         color="tab:red", alpha=0.08, label="bear 狀態")
        axx.plot(df.index, df["exposure"], color=color, lw=0.8)
        axx.set_ylim(0, 1.05)
        axx.set_ylabel(f"曝險 frac={frac:.0%}", fontsize=9)
        axx.legend(fontsize=8); axx.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "main_overview.png", dpi=130)

    fig, ax = plt.subplots(figsize=(12, 5))
    for frac, color in [(0.10, "tab:blue"), (0.25, "tab:green")]:
        df = store_bear[frac]
        ax.plot(df.index, df["exposure"], color=color, lw=1.0,
                label=f"frac={frac:.0%}")
    ax.axhline(PUT_CAP, color="tab:red", lw=1, ls="--",
               label=f"put 停賣閘 {PUT_CAP:.0%}")
    ax.set_ylim(0, 1.05)
    ax.set_title("熊市壓力測試（2021-12~2022-12，全年多在 EMA 下）：曝險軌跡——單調上升檢驗")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS / "bear_exposure.png", dpi=130)

    print("\n輸出：results/exp14_ema_wheel/")


if __name__ == "__main__":
    main()
