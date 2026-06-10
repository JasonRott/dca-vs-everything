"""Deribit DVOL（BTC 選擇權隱含波動率指數）歷史資料抓取與快取。

公開 API：/public/get_volatility_index_data，回傳日線 OHLC（單位：年化 IV %）。
BTC DVOL 自 2021-03-24 起有資料。單次上限 1000 點，自動翻頁。
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_dvol(currency: str = "BTC", start: str = "2021-03-01",
               end: str = "2026-06-11") -> pd.DataFrame:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows = []
    cur_end = end_ms          # API 回傳區間內「最後」1000 點 → 往回翻頁
    while cur_end > start_ms:
        r = requests.get(URL, params={
            "currency": currency, "resolution": "1D",
            "start_timestamp": start_ms, "end_timestamp": cur_end}, timeout=30)
        r.raise_for_status()
        data = r.json()["result"]["data"]
        if not data:
            break
        rows.extend(data)
        first_ts = data[0][0]
        if first_ts <= start_ms or len(data) < 1000:
            break
        cur_end = first_ts - 1
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    return df[~df.index.duplicated(keep="first")]


def load_dvol(currency: str = "BTC", start: str = "2021-03-01",
              end: str = "2026-06-11") -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"DVOL_{currency}_{start}_{end}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    df = fetch_dvol(currency, start, end)
    df.to_csv(cache)
    return df


if __name__ == "__main__":
    df = load_dvol()
    print(df.shape)
    print(df.head(3))
    print(df.tail(3))
