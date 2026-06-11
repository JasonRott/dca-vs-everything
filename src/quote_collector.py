"""每日權利金對照收集器：Deribit 市場價 vs BS 理論價 vs 幣安 DCI 報價。

用途：exp15-C / exp18 落地前的兩週實測——量化「平台抽成」與「模型誤差」。
    python src/quote_collector.py            # 追加一筆當日紀錄到 data/quotes_log.csv

三個數據源：
1. Deribit 公開選擇權鏈（免金鑰）：最近的日到期合約、現價 ±1/2/3% 履約價
   的 call/put 買價與標記價（反向合約報價即為「占現貨比例」）。
2. BS 理論價：即時 DVOL + Black-Scholes（與回測管線同式）。
3. 幣安雙幣理財（DCI）：需 API key（唯讀）。金鑰來源（擇一）：
   - 環境變數 BINANCE_API_KEY / BINANCE_API_SECRET
   - configs/binance_keys.json：{"key": "...", "secret": "..."}（已 gitignore）
   無金鑰時自動跳過該欄。
4. Pionex 欄留空，開通 Dual API 前手動填入。

輸出欄位：date, asset, side, otm_pct, expiry_days,
deribit_bid_pct, deribit_mark_pct, bs_theory_pct,
binance_apr, binance_prem_pct, pionex_apr(手填), spot, dvol
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coin_accum import bs_call
from dual_wheel import bs_put

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "quotes_log.csv"
DERIBIT = "https://www.deribit.com/api/v2"
OTMS = [0.01, 0.02, 0.03]


# ---------------- Deribit 公開資料 ----------------
def deribit_get(path, **params):
    r = requests.get(f"{DERIBIT}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()["result"]


def deribit_chain(ccy: str):
    spot = deribit_get("/public/get_index_price",
                       index_name=f"{ccy.lower()}_usd")["index_price"]
    dvol = deribit_get("/public/get_volatility_index_data",
                       currency=ccy, resolution="1D",
                       start_timestamp=int((time.time() - 5 * 86400) * 1000),
                       end_timestamp=int(time.time() * 1000))["data"][-1][4]
    instrs = deribit_get("/public/get_instruments", currency=ccy,
                         kind="option", expired="false")
    now_ms = time.time() * 1000
    expiries = sorted({i["expiration_timestamp"] for i in instrs})
    nearest = min(e for e in expiries if e > now_ms + 3600_000)
    days = (nearest - now_ms) / 86400_000
    rows = []
    for otm in OTMS:
        for side in ("call", "put"):
            target = spot * (1 + otm) if side == "call" else spot * (1 - otm)
            cands = [i for i in instrs
                     if i["expiration_timestamp"] == nearest
                     and i["option_type"] == side]
            best = min(cands, key=lambda i: abs(i["strike"] - target))
            tk = deribit_get("/public/ticker",
                             instrument_name=best["instrument_name"])
            theo = (bs_call if side == "call" else bs_put)(
                spot, best["strike"], dvol / 100, days / 365) / spot
            rows.append({
                "side": side, "otm_pct": otm * 100,
                "expiry_days": round(days, 2),
                "strike": best["strike"],
                "deribit_bid_pct": (tk.get("best_bid_price") or 0) * 100,
                "deribit_mark_pct": tk["mark_price"] * 100,
                "bs_theory_pct": theo * 100,
            })
    return spot, dvol, rows


# ---------------- 幣安 DCI（需金鑰） ----------------
def load_binance_keys():
    k, s = os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET")
    if k and s:
        return k, s
    f = ROOT / "configs" / "binance_keys.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        return d.get("key"), d.get("secret")
    return None, None


def binance_dci(invest_coin: str, exercised_coin: str, option_type: str):
    """回傳 DCI 產品清單（apr, strikePrice, duration）；無金鑰回 None。"""
    key, secret = load_binance_keys()
    if not key:
        return None
    params = {"optionType": option_type, "investCoin": invest_coin,
              "exercisedCoin": exercised_coin, "pageSize": 100,
              "timestamp": int(time.time() * 1000), "recvWindow": 10000}
    qs = urllib.parse.urlencode(params)
    sig = hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    r = requests.get(f"https://api.binance.com/sapi/v1/dci/product/list?{qs}"
                     f"&signature={sig}",
                     headers={"X-MBX-APIKEY": key}, timeout=20)
    r.raise_for_status()
    return r.json().get("list", [])


def match_dci(products, spot, otm, side):
    """找 duration 最短且履約價最接近目標的產品，回 (apr, prem_pct)。"""
    if not products:
        return None, None
    target = spot * (1 + otm) if side == "call" else spot * (1 - otm)
    best, score = None, None
    for p in products:
        try:
            strike = float(p["strikePrice"])
            dur = int(p["duration"])
        except (KeyError, ValueError):
            continue
        s = (abs(strike - target) / spot, dur)
        if score is None or s < score:
            best, score = p, s
    if best is None:
        return None, None
    apr = float(best["apr"])
    prem_pct = apr * int(best["duration"]) / 365 * 100
    return apr, prem_pct


# ---------------- 主流程 ----------------
def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    out = []
    for ccy, sym in [("BTC", "BTC"), ("ETH", "ETH")]:
        spot, dvol, rows = deribit_chain(ccy)
        dci_call = binance_dci(sym, "USDT", "CALL")   # 止盈寶：持幣賣高
        dci_put = binance_dci("USDT", sym, "PUT")     # 抄底寶：持U買低
        for r in rows:
            prods = dci_call if r["side"] == "call" else dci_put
            apr, prem = match_dci(prods, spot, r["otm_pct"] / 100, r["side"])
            out.append({"date": today, "asset": ccy, **r,
                        "binance_apr": apr, "binance_prem_pct": prem,
                        "pionex_strike": None, "pionex_days": None,
                        "pionex_apr": None, "spot": spot, "dvol": dvol})
    df = pd.DataFrame(out)
    LOG.parent.mkdir(exist_ok=True)
    if LOG.exists():        # 欄位演進時自動對齊舊紀錄
        old = pd.read_csv(LOG)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(LOG, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    cols = ["asset", "side", "otm_pct", "expiry_days", "strike",
            "deribit_bid_pct", "deribit_mark_pct", "bs_theory_pct",
            "binance_apr", "binance_prem_pct"]
    print(df[cols].round(4).to_string(index=False))
    if df["binance_apr"].isna().all():
        print("\n[提示] 未偵測到幣安金鑰——僅記錄 Deribit + 理論價。"
              "設定環境變數 BINANCE_API_KEY/SECRET 或 configs/binance_keys.json"
              "（唯讀權限即可）後重跑。")
    print(f"\n已追加至 {LOG}")


if __name__ == "__main__":
    main()
