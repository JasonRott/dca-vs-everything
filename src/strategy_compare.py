"""通用策略比較模組：資產 DCA × 雙輪選擇權 × 網格，一份設定跑出完整報表。

用法：
    python src/strategy_compare.py --config configs/eth_example.json

設定檔（JSON）結構見 configs/ 範例。輸入四件事：
  1. 資產資料：{"symbol": "ETHUSDT"}（幣安自動抓）或 {"csv": "path.csv"}
     （datetime 索引 + open/high/low/close 欄）
  2. 牛熊基準：{"ema_span": 200, "buffer": 0.03}
  3. 選擇權/網格設定：iv 來源（deribit:ETH / constant:55 / realized:1.1）、
     各策略參數
  4. 投入模式：每月投入金額（閒置資金 5% APY 為框架固定假設）

輸出（out 目錄）：main_window.csv、rolling.csv、overview.png、REPORT.md。
策略集（strategies 列表選用）：cash / dca / grid / dca_cc / dca_puts / dual_wheel
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cc_tuning import run_cc_plus
from coin_accum import HOURLY_R, run_accum
from dca_covered_call import run_dca_cc
from dca_via_puts import run_dca_via_puts
from ema_accum import build_ema
from ema_wheel import run_ema_wheel
from fetch_binance import load_klines
from fetch_deribit import load_dvol
from grid_vs_dca import iter_months, window_metrics

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False


# ---------------- 資料與 IV ----------------
def build_bars(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = cfg["data"]
    if "csv" in d:
        bars = pd.read_csv(d["csv"], index_col=0, parse_dates=True)
        bars.index = pd.to_datetime(bars.index, utc=True)
        bars = bars[["open", "high", "low", "close"]].astype(float)
    else:
        bars = load_klines(d["symbol"], d.get("interval", "1h"),
                           d.get("start", "2018-01-01"),
                           d.get("end", "2026-06-01"))
    daily = bars.resample("1D").agg({"open": "first", "high": "max",
                                     "low": "min", "close": "last"}).dropna()
    return bars, daily


def build_iv(cfg: dict, daily: pd.DataFrame) -> pd.Series:
    """回傳日線 IV（%、年化、shift(1) 防前視）。"""
    spec = cfg.get("iv", "realized:1.1")
    kind, _, arg = spec.partition(":")
    if kind == "deribit":
        return load_dvol(arg or "BTC", "2021-03-01",
                         cfg["data"].get("end", "2026-06-11"))["close"].shift(1)
    if kind == "constant":
        return pd.Series(float(arg), index=daily.index).shift(1)
    if kind == "realized":           # EWMA 已實現波動 × 倍數
        mult = float(arg or 1.1)
        ret = np.log(daily["close"]).diff()
        rv = ret.ewm(halflife=30).std() * np.sqrt(365) * 100 * mult
        return rv.shift(1)
    raise ValueError(f"未知 iv 模式：{spec}")


# ---------------- 策略執行 ----------------
def run_strategies(cfg, bars, ema, iv, contrib):
    out = {}     # name -> (value_series, coin_series|None, extra dict)
    sel = cfg.get("strategies",
                  ["cash", "dca", "grid", "dca_cc", "dca_puts", "dual_wheel"])
    if "cash" in sel:
        pool, vals, idx = 0.0, [], []
        for _, mb in iter_months(bars):
            pool += contrib
            for _ in range(len(mb)):
                pool *= 1 + HOURLY_R
            vals.append(pool)
            idx.append(mb.index[-1])
        out["現金5%"] = (pd.Series(vals, index=pd.DatetimeIndex(idx)),
                        None, {})
    if "dca" in sel:
        _, df = run_dca_cc(bars, 0.0, contrib=contrib)
        out["DCA"] = (df["value"], df["btc"], {})
    if "grid" in sel:
        g = cfg.get("grid", {})
        diag, df = run_accum(bars, g.get("ratio", 0.5), g.get("spacing", 0.03),
                             contrib=contrib, width=g.get("width", 0.15),
                             down_reopen=g.get("down_reopen", "immediate"))
        out["網格"] = (df["value"], df["btc"],
                      {"avg_cost": diag["bucket_avg_cost"]})
    if "dca_cc" in sel:
        c = cfg.get("cc", {})
        diag, df = run_cc_plus(bars, ema, iv, contrib=contrib,
                               inst_rebuy=c.get("inst_rebuy", True),
                               bear_put=c.get("bear_put", False),
                               frac=c.get("frac", 0.25),
                               otm_fix=c.get("otm", 0.03),
                               buffer=cfg.get("regime", {}).get("buffer", 0.03))
        out["DCA+CC"] = (df["value"], df["btc"],
                         {"premium": diag["prem_call"] + diag["prem_put"]})
    if "dca_puts" in sel:
        p = cfg.get("puts", {})
        diag, df = run_dca_via_puts(bars, iv, p.get("otm", 0.02),
                                    contrib=contrib)
        out["抄底寶DCA"] = (df["value"], df["btc"],
                           {"premium": diag["premium"]})
    if "dual_wheel" in sel:
        w = cfg.get("wheel", {})
        diag, df = run_ema_wheel(bars, ema, iv, w.get("frac", 0.10),
                                 otm=w.get("otm", 0.03),
                                 buffer=cfg.get("regime", {}).get("buffer", 0.03),
                                 put_cap=w.get("put_cap", 0.80),
                                 contrib=contrib)
        out["雙輪"] = (df["value"], df["btc"],
                      {"premium": diag["prem_call"] + diag["prem_put"]})
    return out


def metrics_table(results, nm, contrib):
    rows = []
    dqty = None
    if "DCA" in results and results["DCA"][1] is not None:
        dqty = float(results["DCA"][1].iloc[-1])
    for name, (val, coin, extra) in results.items():
        met = window_metrics(val, nm, contrib)
        row = {"strategy": name, "moic": met["moic"],
               "irr_ann": met["irr_ann"], "max_dd": met["max_dd"],
               "calmar": (met["irr_ann"] / abs(met["max_dd"])
                          if met["max_dd"] < 0 else np.nan)}
        if coin is not None and dqty:
            row["coin_ratio"] = float(coin.iloc[-1]) / dqty
        row.update(extra)
        rows.append(row)
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.3f}" if isinstance(v, (int, float, np.floating))
                         and not pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------- 主流程 ----------------
def run(cfg: dict):
    out_dir = Path(cfg.get("out", "results/compare"))
    out_dir.mkdir(parents=True, exist_ok=True)
    contrib = float(cfg.get("contrib", 1000))

    bars, daily = build_bars(cfg)
    ema = build_ema(daily, bars.index,
                    span=cfg.get("regime", {}).get("ema_span", 200))
    iv = build_iv(cfg, daily)

    # 主視窗
    main_start = cfg.get("main_start", str(bars.index[0].date()))
    sub = bars.loc[main_start:]
    nm = sum(1 for _ in iter_months(sub))
    results = run_strategies(cfg, sub, ema, iv, contrib)
    mdf = metrics_table(results, nm, contrib)
    mdf.to_csv(out_dir / "main_window.csv", index=False)

    # 滾動視窗
    rc = cfg.get("rolling", {})
    months, step = rc.get("months", 24), rc.get("step", 3)
    starts = pd.date_range(rc.get("start", str(bars.index[0].date())),
                           rc.get("end", str(bars.index[-1].date())),
                           freq=f"{step}MS", tz="UTC")
    roll = []
    for st in starts:
        s2 = bars.loc[st:st + pd.DateOffset(months=months)]
        nm2 = sum(1 for _ in iter_months(s2))
        if nm2 < months:
            continue
        res2 = run_strategies(cfg, s2, ema, iv, contrib)
        m2 = metrics_table(res2, nm2, contrib).set_index("strategy")
        r = {"start": st}
        for name in m2.index:
            r[f"{name}_moic"] = m2.loc[name, "moic"]
            r[f"{name}_mdd"] = m2.loc[name, "max_dd"]
        roll.append(r)
    rdf = pd.DataFrame(roll)
    rdf.to_csv(out_dir / "rolling.csv", index=False)

    # 勝率摘要（vs DCA）
    winrows = []
    if not rdf.empty and "DCA_moic" in rdf:
        for name in [n for n in results if n != "DCA"]:
            if f"{name}_moic" not in rdf:
                continue
            winrows.append({
                "strategy": name,
                "moic_win_vs_dca": float(
                    (rdf[f"{name}_moic"] > rdf["DCA_moic"]).mean()),
                "mdd_shallower": float(
                    (rdf[f"{name}_mdd"] > rdf["DCA_mdd"]).mean()),
            })
    wdf = pd.DataFrame(winrows)

    # 圖
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (name, (val, _, _)) in enumerate(results.items()):
        axes[0].plot(val.index, val.values, lw=1.1, color=palette[i],
                     label=name)
    cidx = pd.date_range(sub.index[0], periods=nm, freq="MS")
    axes[0].plot(cidx, np.arange(1, nm + 1) * contrib, "k--", lw=1,
                 label="累計投入")
    axes[0].set_title(f"主視窗 {main_start}~（{nm} 個月）：法幣淨值")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    if not rdf.empty:
        for i, name in enumerate(results):
            col = f"{name}_moic"
            if col in rdf:
                axes[1].plot(rdf["start"], rdf[col], "-o", ms=3,
                             color=palette[i], label=name)
        axes[1].axhline(1, color="gray", lw=0.8)
        axes[1].set_title(f"滾動 {months} 個月 MOIC")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "overview.png", dpi=130)

    # REPORT.md
    label = cfg["data"].get("symbol", cfg["data"].get("csv", "asset"))
    px0, px1 = float(sub["open"].iloc[0]), float(sub["close"].iloc[-1])
    report = [
        f"# 策略比較報表：{label}",
        f"\n主視窗：{main_start} 起 {nm} 個月，"
        f"價格 {px0:,.0f} → {px1:,.0f}（{px1/px0-1:+.1%}）。",
        f"月投入 {contrib:,.0f}；IV 來源 `{cfg.get('iv')}`；"
        f"牛熊基準 EMA{cfg.get('regime', {}).get('ema_span', 200)} "
        f"±{cfg.get('regime', {}).get('buffer', 0.03):.0%}。\n",
        "## 主視窗總表\n", md_table(mdf.round(3)),
        f"\n## 滾動 {months} 個月勝率（vs DCA，{len(rdf)} 視窗）\n",
        md_table(wdf.round(2)) if not wdf.empty else "（無滾動資料）",
        "\n## 檔案\n",
        "- main_window.csv / rolling.csv / overview.png",
        "\n> 框架假設：閒置資金 5% APY、現貨費 0.1%、選擇權 1 天期 BS 定價"
        "（無交易費）、月初投入。詳見 repo 主 README 的限制聲明。",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(mdf.round(3).to_string(index=False))
    if not wdf.empty:
        print()
        print(wdf.round(2).to_string(index=False))
    print(f"\n輸出：{out_dir}/（REPORT.md, main_window.csv, rolling.csv, overview.png）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        run(json.load(f))
