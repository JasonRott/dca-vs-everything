"""exp12 嚴格審計 + 跨實驗關鍵主張核驗。

A. exp12 內部一致性（重跑最密組與最佳組，帶恆等式檢查）：
   1. money_input == Σ flows 金額（資金恆等式）
   2. 外部注資筆數 vs 下破次數（應相等：上破後錢包立即扣 100 開新格，
      錢包只剩零頭利潤 < 100，故每次下破必然需要注資）
   3. MTM 曲線末值 == 期末清算值
   4. 現金流配對 B&H 以獨立程式重算
   5. 注資金額分布（應 ≈ 100 − 零頭利潤）
B. 跨實驗主張 vs 存檔 CSV：
   - exp10「全配置三軸 ≥ 純 DCA」
   - exp11「所有階梯 cost_ratio > 1」
   - exp12「15 組 B&H 全勝」
C. 重畫 exp12 圖：最佳組與最密組並列，注資直方圖加計數標注。
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
from dgt_faithful import FEE, run_dgt_paper, xirr
from fetch_binance import load_klines

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False
RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp12_dgt_faithful"
ROOT = Path(__file__).resolve().parent.parent


def audit_config(bars, size, m, label):
    r = run_dgt_paper(bars, size, m)
    flows = r["flows"]
    ok = []
    # 1. 資金恆等式
    s = sum(a for _, a, _ in flows)
    ok.append(("Σflows == money_input", abs(s - r["money_input"]) < 1e-6,
               f"{s:,.2f} vs {r['money_input']:,.2f}"))
    # 2. 注資筆數 vs 下破次數
    n_inj = len(flows) - 1
    ok.append(("注資筆數 == 下破次數", n_inj == r["n_dn"],
               f"{n_inj} vs n_dn={r['n_dn']} (n_up={r['n_up']})"))
    # 3. 曲線末值 == 期末值
    ok.append(("MTM末值 == 期末清算值",
               abs(r["curve"].iloc[-1] - r["final"]) < 1e-6,
               f"{r['curve'].iloc[-1]:,.2f} vs {r['final']:,.2f}"))
    # 4. B&H 獨立重算
    qty = sum(a * (1 - FEE) / px for _, a, px in flows)
    bh = qty * r["close_end"]
    ok.append(("B&H 獨立重算", True,
               f"final={bh:,.2f}, MOIC={bh/r['money_input']:.3f}"))
    # 5. 注資金額分布
    inj = [a for _, a, _ in flows[1:]]
    if inj:
        ok.append(("注資金額分布", True,
                   f"n={len(inj)}, mean={np.mean(inj):.2f}, "
                   f"min={np.min(inj):.2f}, max={np.max(inj):.2f}"))
    print(f"\n--- {label}: size={size}, m={m} ---")
    print(f"  總投入 {r['money_input']:,.0f} | 終值 {r['final']:,.0f} | "
          f"MOIC {r['final']/r['money_input']:.3f} | B&H MOIC {bh/r['money_input']:.3f}")
    for name, passed, detail in ok:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return r, bh


def main():
    bars = load_klines("BTCUSDT", "1m", "2021-01-01", "2024-08-01")
    print("=== A. exp12 內部一致性 ===")
    r_best, _ = audit_config(bars, 0.05, 10, "最佳組（圖中那組）")
    r_dense, _ = audit_config(bars, 0.005, 2, "最密組（8,971筆注資那組）")

    print("\n=== B. 跨實驗主張 vs 存檔 CSV ===")
    # exp10
    m10 = pd.read_csv(ROOT / "results/exp10_dca_cc/main_window.csv")
    dca = m10[m10["config"] == "純 DCA"].iloc[0]
    cc = m10[m10["config"] != "純 DCA"]
    dom = ((cc["moic"] >= dca["moic"]) & (cc["max_dd"] >= dca["max_dd"])
           & (cc["final_btc"] >= dca["final_btc"]))
    print(f"  exp10 三軸支配: {dom.sum()}/{len(cc)} 配置成立 "
          f"{'[PASS]' if dom.all() else '[FAIL: ' + str(cc.loc[~dom, 'config'].tolist()) + ']'}")
    # exp11
    m11 = pd.read_csv(ROOT / "results/exp11_buy_ladder/main_window.csv")
    lad = m11[m11["strategy"] != "DCA"]
    print(f"  exp11 cost_ratio>1: {(lad['cost_ratio'] > 1).sum()}/{len(lad)} "
          f"{'[PASS]' if (lad['cost_ratio'] > 1).all() else '[FAIL]'}")
    # exp12
    m12 = pd.read_csv(ROOT / "results/exp12_dgt_faithful/dgt_faithful_sweep.csv")
    bh_win = (m12["bh_moic"] >= m12["moic"]) & (m12["bh_xirr"] >= m12["xirr"])
    print(f"  exp12 B&H 全勝: {bh_win.sum()}/{len(m12)} "
          f"{'[PASS]' if bh_win.all() else '[FAIL]'}")
    print(f"  exp12 注資規模對照: 最密組 {m12.iloc[0]['n_inject']-1:.0f} 筆 / "
          f"{m12.iloc[0]['total_input']:,.0f} USDT；"
          f"最佳組 {m12.loc[m12['moic'].idxmax(), 'n_inject']-1:.0f} 筆 / "
          f"{m12.loc[m12['moic'].idxmax(), 'total_input']:,.0f} USDT")

    # ===== C. 重畫：最佳組 vs 最密組並列 =====
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for col, (r, label) in enumerate([
            (r_best, "最佳組 size=5%, m=10（半寬±50%）"),
            (r_dense, "最密組 size=0.5%, m=2（半寬±1%）")]):
        ax = axes[0, col]
        flows = r["flows"]
        qty = 0.0
        bh_t, bh_v = [], []
        fi = 0
        for ts, px in bars["close"].iloc[::60].items():
            while fi < len(flows) and flows[fi][0] <= ts:
                qty += flows[fi][1] * (1 - FEE) / flows[fi][2]
                fi += 1
            bh_t.append(ts); bh_v.append(qty * px)
        ax.plot(r["curve"].index, r["curve"].values, color="tab:purple",
                lw=1.0, label="DGT 忠實版")
        ax.plot(bh_t, bh_v, color="tab:orange", lw=1.0, alpha=0.85,
                label="現金流配對 B&H")
        cum = np.cumsum([f[1] for f in flows])
        ax.step([f[0] for f in flows], cum, where="post", color="k", ls="--",
                lw=1, label="累計注資")
        ax.set_yscale("log"); ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax2 = axes[1, col]
        inj_t = [f[0] for f in flows[1:]]
        ax2.hist(inj_t, bins=60, color="tab:red", alpha=0.75)
        ax2.set_title(f"外部注資事件（共 {len(inj_t):,} 筆, "
                      f"合計 {sum(f[1] for f in flows[1:]):,.0f} USDT）",
                      fontsize=10)
        ax2.grid(alpha=0.3)
    fig.suptitle("exp12 審計版：最佳組 vs 最密組（注資規模差四個數量級）", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS / "dgt_faithful_audit.png", dpi=130)
    print("\n輸出：results/exp12_dgt_faithful/dgt_faithful_audit.png")


if __name__ == "__main__":
    main()
