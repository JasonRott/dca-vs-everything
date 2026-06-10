"""等比網格回測引擎。

模型（Pionex / 幣安現貨網格的標準機制）：
- 區間 [L, H] 切成 n 格，價位 levels[i] = L * (H/L)**(i/n)，i = 0..n。
- 第 j 格（j = 0..n-1）：在 levels[j] 掛買單、levels[j+1] 掛賣單，
  每格分配等額報價貨幣 Q = 本金 / n。
- 啟動時價格 p0：買價 >= p0 的格（限價買單會立即成交）改以市價 p0 建立底倉，
  賣單掛在該格上緣；包含 p0 的格與其下方的格保持現金、等待買入。
- 格內狀態機：現金 --(價格下穿 levels[j] 買入)--> 持幣 --(上穿 levels[j+1] 賣出)--> 現金。
- 棒內路徑近似：收盤 >= 開盤 時走 開->低->高->收，否則 開->高->低->收，
  逐段依序觸發穿越的格線。

費用：成交額 × fee（買入時實得幣量打折、賣出時實得現金打折）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class GridResult:
    final_value: float          # 期末清算後淨值（報價貨幣）
    n_buys: int
    n_sells: int
    fees_paid: float
    grid_profit: float          # 已實現配對利潤（賣出收入 - 對應買入成本）
    exposure: pd.Series = field(repr=False)   # 每棒收盤的持幣市值占比
    value_curve: pd.Series = field(repr=False)  # 每棒收盤的逐棒淨值


class GeometricGridBot:
    def __init__(self, low: float, high: float, n_grids: int,
                 capital: float, fee: float = 0.001):
        assert high > low > 0 and n_grids >= 1
        self.levels = low * (high / low) ** (np.arange(n_grids + 1) / n_grids)
        self.n = n_grids
        self.fee = fee
        self.Q = capital / n_grids       # 每格報價貨幣額度
        self.cash = capital
        self.has_coin = np.zeros(n_grids, dtype=bool)
        self.qty = np.zeros(n_grids)     # 各格持幣量
        self.cost = np.zeros(n_grids)    # 各格買入成本（報價貨幣）
        self.n_buys = 0
        self.n_sells = 0
        self.fees_paid = 0.0
        self.grid_profit = 0.0
        self._started = False
        self._last_price = np.nan

    # ---- 內部動作 -------------------------------------------------
    def _buy_slot(self, j: int, price: float):
        spend = self.Q
        self.cash -= spend
        fee_amt = spend * self.fee
        self.qty[j] = (spend - fee_amt) / price
        self.cost[j] = spend
        self.has_coin[j] = True
        self.fees_paid += fee_amt
        self.n_buys += 1

    def _sell_slot(self, j: int, price: float):
        gross = self.qty[j] * price
        fee_amt = gross * self.fee
        self.cash += gross - fee_amt
        self.grid_profit += (gross - fee_amt) - self.cost[j]
        self.fees_paid += fee_amt
        self.qty[j] = 0.0
        self.has_coin[j] = False
        self.n_sells += 1

    def start(self, p0: float, initial_position: bool = True):
        """啟動：買價 >= p0 的格以市價 p0 建底倉。

        initial_position=False 時不建底倉（純掛單）：所有格保持現金，
        價格下穿各格買價時才買入——上半部格線只在價格回落穿越時成交。
        """
        if initial_position:
            for j in range(self.n):
                if self.levels[j] >= p0:
                    self._buy_slot(j, p0)
        self._started = True
        self._last_price = p0

    def _move_to(self, price: float):
        """價格從 _last_price 單調移動到 price，觸發沿途格線。"""
        prev = self._last_price
        if price < prev:    # 下行：觸發買單（由高至低）
            for j in range(self.n - 1, -1, -1):
                g = self.levels[j]
                if price <= g < prev and not self.has_coin[j]:
                    self._buy_slot(j, g)
        elif price > prev:  # 上行：觸發賣單（由低至高）
            for j in range(self.n):
                g = self.levels[j + 1]
                if prev < g <= price and self.has_coin[j]:
                    self._sell_slot(j, g)
        self._last_price = price

    # ---- 對外介面 -------------------------------------------------
    @property
    def total_qty(self) -> float:
        return float(self.qty.sum())

    def move_to(self, price: float):
        """外部驅動模式：價格單調移動到 price（需先 start）。"""
        assert self._started
        self._move_to(price)

    def run(self, bars: pd.DataFrame) -> GridResult:
        """bars: 含 open/high/low/close 的 DataFrame。啟動價 = 第一棒開盤。"""
        if not self._started:
            self.start(float(bars["open"].iloc[0]))
        expo = np.empty(len(bars))
        vals = np.empty(len(bars))
        for i, (o, h, l, c) in enumerate(
                bars[["open", "high", "low", "close"]].itertuples(index=False)):
            path = (o, l, h, c) if c >= o else (o, h, l, c)
            for p in path:
                self._move_to(p)
            coin_val = float(self.qty.sum()) * c
            vals[i] = self.cash + coin_val
            expo[i] = coin_val / vals[i] if vals[i] > 0 else 0.0
        final = self.liquidate(float(bars["close"].iloc[-1]))
        return GridResult(
            final_value=final, n_buys=self.n_buys, n_sells=self.n_sells,
            fees_paid=self.fees_paid, grid_profit=self.grid_profit,
            exposure=pd.Series(expo, index=bars.index),
            value_curve=pd.Series(vals, index=bars.index),
        )

    def liquidate(self, price: float) -> float:
        """市價賣出全部持幣，回傳淨值。"""
        total_qty = float(self.qty.sum())
        if total_qty > 0:
            gross = total_qty * price
            fee_amt = gross * self.fee
            self.cash += gross - fee_amt
            self.fees_paid += fee_amt
        self.qty[:] = 0.0
        self.has_coin[:] = False
        return self.cash
