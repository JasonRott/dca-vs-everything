"""幣安 K 線資料抓取與本地快取。

公開 REST API（不需金鑰）：GET /api/v3/klines，單次上限 1000 根。
快取為 CSV，存於 data/，重跑時直接讀快取。
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_base", "taker_quote", "ignore",
]


def fetch_klines(symbol: str, interval: str, start: str, end: str,
                 pause: float = 0.15) -> pd.DataFrame:
    """抓取 [start, end) 區間的 K 線（UTC），自動翻頁。"""
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows = []
    cur = start_ms
    while cur < end_ms:
        resp = requests.get(BASE_URL, params={
            "symbol": symbol, "interval": interval,
            "startTime": cur, "endTime": end_ms - 1, "limit": 1000,
        }, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][6] + 1  # 最後一根的 close_time + 1ms
        time.sleep(pause)

    df = pd.DataFrame(rows, columns=KLINE_COLS)
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_klines(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    """讀取快取，無快取則抓取後寫入。"""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"{symbol}_{interval}_{start}_{end}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    df = fetch_klines(symbol, interval, start, end)
    df.to_csv(cache)
    return df


if __name__ == "__main__":
    df = load_klines("BTCUSDT", "1h", "2018-01-01", "2026-06-01")
    print(df.shape)
    print(df.head(2))
    print(df.tail(2))
