import requests
import pandas as pd
from datetime import datetime, timezone


def to_ms(time_str):
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def get_future_klines(symbol, interval, start, end):

    url = "https://fapi.binance.com/fapi/v1/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": to_ms(start),
        "endTime": to_ms(end),
        "limit": 1000
    }

    r = requests.get(url, params=params)
    r.raise_for_status()

    data = r.json()

    df = pd.DataFrame(
        data,
        columns=[
            "openTime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "closeTime",
            "quoteVolume",
            "trades",
            "takerBuyBase",
            "takerBuyQuote",
            "ignore"
        ]
    )

    df["openTime"] = pd.to_datetime(
        df["openTime"],
        unit="ms"
    )

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[c] = df[c].astype(float)

    return df


for symbol in ["BTCUSDT", "ETHUSDT"]:

    df = get_future_klines(
        symbol,
        "3d",
        "2021-01-01 00:00:00",
        "2026-08-26 00:00:00"
    )

    df.to_parquet(
        f"{symbol}_3d.parquet",
        index=False
    )

    print(symbol, len(df))