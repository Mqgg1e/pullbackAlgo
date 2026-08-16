"""
Step 1: Swing point detection - ZigZag method.

Replaces the earlier fixed-window fractal approach. That approach flagged
every local wiggle as a pivot independently for highs and lows, which meant
in choppy stretches it produced dense, noisy points with no guarantee that
consecutive high/low pivots were separated by a meaningful move - occasionally
even producing a "high" pivot sitting at a lower price than a nearby "low"
pivot once the timeline got busy.

ZigZag fixes this structurally: it tracks a single running extreme at a time
and only confirms a pivot once price has reversed away from that extreme by
at least a minimum threshold. Because of that, the resulting pivot sequence
always alternates high/low/high/low... and every pivot represents a real,
minimum-sized move - not just local noise.

Input is expected to be a pandas DataFrame with at least the columns:
    openTime, high, low, close
(the schema produced by main.py / stored in the data/*.parquet files).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class SearchState(Enum):
    SEARCHING_HIGH = "searchingHigh"  # tracking a rising run, looking for the top
    SEARCHING_LOW = "searchingLow"    # tracking a falling run, looking for the bottom


@dataclass
class SwingPoint:
    barIndex: int
    openTime: pd.Timestamp
    price: float
    isHigh: bool  # True = swing high, False = swing low


def computeAtr(klines: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, used as a volatility-adaptive reversal threshold."""
    high = klines["high"]
    low = klines["low"]
    prevClose = klines["close"].shift(1)

    trueRange = pd.concat([
        high - low,
        (high - prevClose).abs(),
        (low - prevClose).abs(),
    ], axis=1).max(axis=1)

    return trueRange.rolling(period, min_periods=1).mean()


def detectSwingPoints(
    klines: pd.DataFrame,
    minDeviation: float = 0.05,
    useAtr: bool = False,
    atrPeriod: int = 14,
    atrMultiple: float = 2.0,
) -> list[SwingPoint]:
    """
    Detect swing points via ZigZag.

    minDeviation: minimum reversal size as a fraction of price (e.g. 0.05 = 5%).
                  Used when useAtr is False.
    useAtr:       if True, use `atrMultiple * ATR(atrPeriod)` as the reversal
                  threshold instead of a flat percentage - adapts to each
                  symbol/period's own volatility rather than a fixed number.
    """
    highs = klines["high"].to_numpy()
    lows = klines["low"].to_numpy()
    openTimes = klines["openTime"].to_numpy()
    barCount = len(klines)

    atr = computeAtr(klines, atrPeriod).to_numpy() if useAtr else None

    def thresholdAt(referenceIndex: int, referencePrice: float) -> float:
        if useAtr:
            return atr[referenceIndex] * atrMultiple
        return referencePrice * minDeviation

    swings: list[SwingPoint] = []

    state: Optional[SearchState] = None
    candidateHighPrice = highs[0]
    candidateHighIndex = 0
    candidateLowPrice = lows[0]
    candidateLowIndex = 0

    for i in range(1, barCount):
        if state in (None, SearchState.SEARCHING_HIGH):
            # Check reversal FIRST, using the candidate as it stood before this
            # bar. Only if no reversal fires does this bar's own high get to
            # extend the candidate. This ordering is what guarantees a single
            # bar can never both set a brand-new extreme AND confirm a reversal
            # against that same, just-set extreme in one step.
            dropFromHigh = candidateHighPrice - lows[i]
            if dropFromHigh >= thresholdAt(candidateHighIndex, candidateHighPrice):
                swings.append(SwingPoint(
                    barIndex=candidateHighIndex,
                    openTime=pd.Timestamp(openTimes[candidateHighIndex]),
                    price=float(candidateHighPrice),
                    isHigh=True,
                ))
                state = SearchState.SEARCHING_LOW
                candidateLowPrice = lows[i]
                candidateLowIndex = i
                continue

            if highs[i] > candidateHighPrice:
                candidateHighPrice = highs[i]
                candidateHighIndex = i

        if state in (None, SearchState.SEARCHING_LOW):
            riseFromLow = highs[i] - candidateLowPrice
            if riseFromLow >= thresholdAt(candidateLowIndex, candidateLowPrice):
                swings.append(SwingPoint(
                    barIndex=candidateLowIndex,
                    openTime=pd.Timestamp(openTimes[candidateLowIndex]),
                    price=float(candidateLowPrice),
                    isHigh=False,
                ))
                state = SearchState.SEARCHING_HIGH
                candidateHighPrice = highs[i]
                candidateHighIndex = i
                continue

            if lows[i] < candidateLowPrice:
                candidateLowPrice = lows[i]
                candidateLowIndex = i

    return swings


def swingPointsToFrame(swings: list[SwingPoint]) -> pd.DataFrame:
    """Convenience helper to convert the swing point list into a DataFrame,
    handy for inspection, CSV export, or feeding downstream steps."""
    return pd.DataFrame([
        {
            "barIndex": s.barIndex,
            "openTime": s.openTime,
            "price": s.price,
            "isHigh": s.isHigh,
        }
        for s in swings
    ])


if __name__ == "__main__":
    klines = pd.read_parquet("data/MUUSDT_1d.parquet")

    swings = detectSwingPoints(klines, useAtr=True, atrPeriod=14, atrMultiple=2.0)

    swingFrame = swingPointsToFrame(swings)
    print(swingFrame)