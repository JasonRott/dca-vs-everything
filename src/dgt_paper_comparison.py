"""exp12b：DGT 檢驗修正——忠實重現 vs 論文原圖數值 vs 正確基準。

對照來源（literature/figures/，取自 arXiv HTML 原圖）：
- DGTBTC.png：論文 BTC 各 grid size 的平均 IRR 長條（28–36%）、
  Buy-and-Hold IRR ≈ 22.3%（僅持有期初本金）
- irrbtc.png / irreth.png：等高線圖；「60–70%」為 ETH 峰值角落

本圖把三者放在同一座標：
1. 論文 DGT IRR（目測自原圖長條，±1pp）
2. 忠實重現的論文式 IRR（exp12 sweep，按 size 對 m∈{2,5,10} 取平均）
3. 現金流配對 B&H 的 XIRR（同錢、同時點直接買入持有）——正確基準

結論視覺化：論文值與正確 XIRR 量級相符（數值可信）；
但無論採信哪一組，現金流配對基準（3）都高於 DGT——關鍵在基準選擇。
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

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False
RESULTS = Path(__file__).resolve().parent.parent / "results" / "exp12_dgt_faithful"

# 論文 DGTBTC.png 長條目測值（各 grid size 對 m 的平均 IRR, %）
PAPER_BTC_IRR = {0.005: 27.9, 0.01: 29.0, 0.02: 30.5, 0.05: 33.6, 0.10: 35.9}
PAPER_BH_IRR = 22.3          # 論文 B&H（僅期初本金）


def main():
    sweep = pd.read_csv(RESULTS / "dgt_faithful_sweep.csv")
    agg = sweep.groupby("size")[["paper_irr", "xirr", "bh_xirr"]].mean() * 100
    agg = agg.loc[list(PAPER_BTC_IRR)]
    paper_vals = [PAPER_BTC_IRR[s] for s in agg.index]

    out = agg.assign(paper_reported=paper_vals)
    out.to_csv(RESULTS / "dgt_verdict_revision.csv")
    print(out.round(1).to_string())

    x = np.arange(len(agg))
    w = 0.27
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(x - w, paper_vals, w, color="#c8b8e8", edgecolor="k", lw=0.5,
           label="論文報告值（目測自原圖, ±1pp）")
    ax.bar(x, agg["paper_irr"], w, color="tab:purple", edgecolor="k", lw=0.5,
           label="忠實重現（官方引擎+論文式IRR）")
    ax.bar(x + w, agg["bh_xirr"], w, color="tab:orange", edgecolor="k", lw=0.5,
           label="現金流配對 B&H 的 XIRR（正確基準）")
    ax.axhline(PAPER_BH_IRR, color="tab:red", lw=1.4, ls="--",
               label=f"論文的 B&H 基準（僅期初本金, {PAPER_BH_IRR:.0f}%）")
    ax.set_xticks(x)
    ax.set_xticklabels([f"size={s:.1%}" for s in agg.index])
    ax.set_ylabel("年化 IRR (%)")
    ax.set_title("exp12b：DGT 檢驗——論文值接近正確 XIRR（數值可信），"
                 "關鍵在基準選擇（紅線基準未含後續注資）\n"
                 "BTC/USDT，2021-01~2024-07，各 grid size 對 m=2/5/10 平均")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS / "dgt_verdict_revision.png", dpi=130)
    print("\n輸出：dgt_verdict_revision.png, dgt_verdict_revision.csv")


if __name__ == "__main__":
    main()
