from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import requests
import pandas as pd
import numpy as np
import time
import os

app = FastAPI()

SYMBOLS_CACHE = {"data": [], "ts": 0}

TF_MAP_BINANCE = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"
}
TF_MAP_BITFINEX = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "4h", "1d": "1D", "1w": "1W"
}

def clean_val(val, default=0.0):
    if val is None or np.isnan(val) or np.isinf(val):
        return float(default)
    return float(val)

@app.get("/api/symbols")
def get_symbols():
    now = time.time()
    if SYMBOLS_CACHE["data"] and (now - SYMBOLS_CACHE["ts"] < 3600):
        return SYMBOLS_CACHE["data"]

    special = [
        {"symbol": "BTCUSDLONGS", "source": "BITFINEX", "desc": "Bitfinex BTC Margin Longs"},
        {"symbol": "BTCUSDSHORTS", "source": "BITFINEX", "desc": "Bitfinex BTC Margin Shorts"},
        {"symbol": "ETHUSDLONGS", "source": "BITFINEX", "desc": "Bitfinex ETH Margin Longs"},
        {"symbol": "ETHUSDSHORTS", "source": "BITFINEX", "desc": "Bitfinex ETH Margin Shorts"},
    ]
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=8)
        tickers = r.json()
        usdt_pairs = [
            t for t in tickers
            if isinstance(t, dict) and t.get("symbol", "").endswith("USDT") and float(t.get("quoteVolume", 0)) > 0
        ]
        usdt_pairs.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
        binance_list = [
            {"symbol": t["symbol"], "source": "BINANCE", "price": float(t["lastPrice"]), "chg": float(t["priceChangePercent"])}
            for t in usdt_pairs
        ]
        SYMBOLS_CACHE["data"] = special + binance_list
        SYMBOLS_CACHE["ts"] = now
    except Exception:
        if not SYMBOLS_CACHE["data"]:
            SYMBOLS_CACHE["data"] = special + [{"symbol": "BTCUSDT", "source": "BINANCE"}]
    return SYMBOLS_CACHE["data"]


def fetch_ohlcv(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # Manejo de Longs y Shorts de Bitfinex (pos.size en la API de Stats)
    if symbol in ["BTCUSDLONGS", "BTCUSDSHORTS", "ETHUSDLONGS", "ETHUSDSHORTS"]:
        coin = "tBTCUSD" if symbol.startswith("BTC") else "tETHUSD"
        side = "long" if symbol.endswith("LONGS") else "short"
        url = f"https://api-pub.bitfinex.com/v2/stats1/pos.size:1m:{coin}:{side}/hist?limit={limit}&sort=-1"
        
        try:
            r = requests.get(url, headers=headers, timeout=8)
            raw = r.json()
            if not isinstance(raw, list) or (len(raw) > 0 and not isinstance(raw[0], list)):
                raise ValueError("Respuesta inválida de Bitfinex")
            
            raw.reverse()
            rows = []
            for item in raw:
                if len(item) >= 2:
                    t = int(item[0] // 1000)
                    v = float(item[1])
                    # Mapeamos el valor a OHLC para que sea compatible con las velas e indicadores
                    rows.append([t, v, v, v, v, 1.0])
            
            df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
        except Exception as e:
            print(f"Error consultando Bitfinex: {e}")
            df = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
            
    # Manejo normal de pares de Binance
    else:
        bn_tf = TF_MAP_BINANCE.get(interval, "1h")
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={bn_tf}&limit={limit}"
        raw = requests.get(url, headers=headers, timeout=8).json()
        df = pd.DataFrame(raw).iloc[:, :6]
        df.columns = ["time", "open", "high", "low", "close", "volume"]
        df["time"] = (df["time"] // 1000).astype(int)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def xsa_pine(src: np.ndarray, length: int, wei: int) -> np.ndarray:
    n = len(src)
    out = np.zeros(n)
    sumf = 0.0
    for i in range(n):
        val = 0.0 if np.isnan(src[i]) else src[i]
        prev_len = 0.0 if (i < length or np.isnan(src[i - length])) else src[i - length]
        sumf = sumf - prev_len + val
        ma = sumf / length if i >= length - 1 else val
        out[i] = ma if i == 0 else (val * wei + out[i - 1] * (length - wei)) / length
    return out


@app.get("/api/chart")
def get_chart_data(symbol: str = Query("BTCUSDT"), interval: str = Query("1h")):
    df = fetch_ohlcv(symbol.upper(), interval, limit=500)
    n = len(df)

    # 1. EMAs (10, 55, 100, 200)
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean().bfill()
    df["ema55"] = df["close"].ewm(span=55, adjust=False).mean().bfill()
    df["ema100"] = df["close"].ewm(span=100, adjust=False).mean().bfill()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean().bfill()

    # 2. Squeeze Momentum (LazyBear linreg 20)
    hh20 = df["high"].rolling(20, min_periods=1).max()
    ll20 = df["low"].rolling(20, min_periods=1).min()
    sma20 = df["close"].rolling(20, min_periods=1).mean()
    delta = (df["close"] - (((hh20 + ll20) / 2.0 + sma20) / 2.0)).fillna(0.0)

    x = np.arange(20)
    x_mean = x.mean()
    x_dev = x - x_mean
    denom = np.sum(x_dev ** 2)

    def calc_linreg(y):
        y_mean = np.mean(y)
        slope = np.sum(x_dev * (y - y_mean)) / denom
        return y_mean + slope * (19 - x_mean)

    df["sz"] = delta.rolling(20).apply(calc_linreg, raw=True).fillna(0.0)

    # 3. ADX (14, 14)
    up = df["high"].diff().fillna(0.0)
    down = (-df["low"].diff()).fillna(0.0)
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0.0)

    tr_rma = rma(tr, 14).replace(0, 1e-9)
    di_plus = 100.0 * rma(pd.Series(plus_dm), 14) / tr_rma
    di_minus = 100.0 * rma(pd.Series(minus_dm), 14) / tr_rma
    dx = (100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, 1.0)).fillna(0.0)
    df["adx"] = rma(dx, 14).fillna(0.0)

    # 4. Whale Detector (Blackcat: 🐳 y ❗)
    low_arr = df["low"].values
    high_arr = df["high"].values
    low_prev = np.roll(low_arr, 1); low_prev[0] = low_arr[0]
    high_prev = np.roll(high_arr, 1); high_prev[0] = high_arr[0]

    num_w = xsa_pine(np.abs(low_arr - low_prev), 3, 1)
    den_w = np.maximum(xsa_pine(np.maximum(low_arr - low_prev, 0.0), 3, 1), 1e-9)
    var3 = (num_w / den_w) * 100.0
    var4 = pd.Series(var3 * 10.0).ewm(span=3, adjust=False).mean().values
    var5 = df["low"].rolling(30, min_periods=1).min().values
    var6 = pd.Series(var4).rolling(30, min_periods=1).max().values
    raw_v8 = np.where(low_arr <= var5, (var4 + var6 * 2.0) / 2.0, 0.0)
    var8 = pd.Series(raw_v8).ewm(span=3, adjust=False).mean().fillna(0.0).values / 618.0

    num_a = xsa_pine(np.abs(high_arr - high_prev), 3, 1)
    den_a = np.maximum(xsa_pine(np.maximum(high_arr - high_prev, 0.0), 3, 1), 1e-9)
    var31 = (num_a / den_a) * 100.0
    var41 = pd.Series(var31 * 10.0).ewm(span=3, adjust=False).mean().values
    var51 = df["high"].rolling(30, min_periods=1).max().values
    var61 = pd.Series(var41).rolling(30, min_periods=1).min().values
    raw_v81 = np.where(high_arr >= var51, (var41 + var61 * 2.0) / 2.0, 0.0)
    var81 = pd.Series(raw_v81).ewm(span=3, adjust=False).mean().fillna(0.0).values / 618.0

    # 5. Señales Whale Detector únicamente
    times = df["time"].values
    markers = []

    for i in range(2, n):
        p = i - 1
        t_bar = int(times[p])

        whale_pump = (var8[p] > 0) and (var8[p] > var8[p - 1]) and (var8[p] > var8[i])
        whale_inv = (var81[p] > 0) and (var81[p] > var81[p - 1]) and (var81[p] > var81[i])

        if whale_pump:
            markers.append({"time": t_bar, "position": "belowBar", "color": "#38bdf8", "shape": "circle", "text": "🐳"})
        elif whale_inv:
            markers.append({"time": t_bar, "position": "aboveBar", "color": "#f97316", "shape": "circle", "text": "❗"})

    markers.sort(key=lambda m: m["time"])

    # 6. Volume Profile & POC (Últimas 200 velas)
    vp_df = df.tail(200)
    p_min = float(vp_df["low"].min())
    p_max = float(vp_df["high"].max())
    rows = 80
    step = (p_max - p_min) / rows if p_max > p_min else 1.0
    vol_bins = np.zeros(rows)

    for _, row in vp_df.iterrows():
        b_low, b_high, b_vol = row["low"], row["high"], row["volume"]
        idx_start = max(0, min(rows - 1, int((b_low - p_min) / step)))
        idx_end = max(0, min(rows - 1, int((b_high - p_min) / step)))
        span = max(1, idx_end - idx_start + 1)
        vol_bins[idx_start:idx_end + 1] += b_vol / span

    poc_idx = int(np.argmax(vol_bins))
    poc_price = clean_val(p_min + (poc_idx + 0.5) * step)
    max_bin_vol = float(np.max(vol_bins)) if np.max(vol_bins) > 0 else 1.0
    vp_histogram = [
        {"price": clean_val(p_min + (i + 0.5) * step), "pct": clean_val(vol_bins[i] / max_bin_vol)}
        for i in range(rows)
    ]

    # 7. Panel Inferior: Squeeze + ADX (Nivel 23 = 0) abajo | RSI arriba en Y
    ni = float(df["sz"].abs().max())
    if ni <= 0 or np.isnan(ni):
        ni = 1.0
    scale_factor = ni / 75.0

    sz_prev = df["sz"].shift(1).fillna(0.0)
    sc1, sc2 = df["sz"] >= 0, df["sz"] < 0
    sc3, sc4 = df["sz"] >= sz_prev, df["sz"] < sz_prev
    df["sqz_color"] = np.select(
        [sc1 & sc3, sc1 & sc4, sc2 & sc4, sc2 & sc3],
        ["#00FF00", "#008000", "#FF0000", "#800000"],
        default="#808080"
    )

    df["adx_scaled"] = ((df["adx"] - 23.0) * scale_factor * 2.0).fillna(0.0)

    rsi_apart = 115.0
    delta_c = df["close"].diff().fillna(0.0)
    gain = rma(delta_c.clip(lower=0), 14)
    loss = rma(-delta_c.clip(upper=0), 14).replace(0, 1e-9)
    df["rsi"] = (100.0 - (100.0 / (1.0 + gain / loss))).fillna(50.0)
    df["rsi_y"] = ((df["rsi"] + rsi_apart) * scale_factor).fillna(0.0)

    rsi_b70 = clean_val((70.0 + rsi_apart) * scale_factor)
    rsi_b50 = clean_val((50.0 + rsi_apart) * scale_factor)
    rsi_b30 = clean_val((30.0 + rsi_apart) * scale_factor)

    candles, ema10, ema55, ema100, ema200 = [], [], [], [], []
    sqz_list, adx_list, rsi_list = [], [], []

    for i in range(n):
        t = int(times[i])
        candles.append({
            "time": t,
            "open": clean_val(df["open"].iloc[i]),
            "high": clean_val(df["high"].iloc[i]),
            "low": clean_val(df["low"].iloc[i]),
            "close": clean_val(df["close"].iloc[i])
        })
        ema10.append({"time": t, "value": clean_val(df["ema10"].iloc[i])})
        ema55.append({"time": t, "value": clean_val(df["ema55"].iloc[i])})
        ema100.append({"time": t, "value": clean_val(df["ema100"].iloc[i])})
        ema200.append({"time": t, "value": clean_val(df["ema200"].iloc[i])})

        sqz_list.append({
            "time": t,
            "value": clean_val(df["sz"].iloc[i]),
            "color": str(df["sqz_color"].iloc[i])
        })
        adx_list.append({
            "time": t,
            "value": clean_val(df["adx_scaled"].iloc[i])
        })
        rsi_list.append({
            "time": t,
            "value": clean_val(df["rsi_y"].iloc[i])
        })

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "is_bitfinex": symbol.upper() in ["BTCUSDLONGS", "BTCUSDSHORTS", "ETHUSDLONGS", "ETHUSDSHORTS"],
        "candles": candles,
        "ema10": ema10,
        "ema55": ema55,
        "ema100": ema100,
        "ema200": ema200,
        "poc": poc_price,
        "vp": vp_histogram,
        "markers": markers,
        "squeeze": sqz_list,
        "adx": adx_list,
        "rsi": rsi_list,
        "rsi_bands": {"b70": rsi_b70, "b50": rsi_b50, "b30": rsi_b30}
    }

# Servir archivos estáticos del frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

# Arranque compatible con Render ($PORT) y ejecución local directa
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
