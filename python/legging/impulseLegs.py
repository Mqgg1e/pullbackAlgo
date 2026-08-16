"""
Step 2: Impulse leg identification.

A single swing-to-swing move being "big" isn't enough to call it an impulse
leg - a strong trend is really a *run* of swing points that keeps making
progress: higher highs and higher lows for an uptrend, lower lows and lower
highs for a downtrend. The run ends the instant a swing point breaks that
structure (e.g. a lower low shows up inside an uptrend) - and that break
point is exactly where a pullback (simple or complex) begins.

So this step does two things:
  1. buildTrendRuns   - group consecutive swings into maximal HH/HL or LL/LH runs
  2. detectImpulseLegs - keep only the runs whose total move is large enough,
                         relative to volatility (ATR), to count as a genuine
                         impulse leg rather than noise
"""

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from ..zigzag.swingPoints import SwingPoint, computeAtr


class TrendDirection(Enum):
    UP = "up"
    DOWN = "down"


@dataclass
class ImpulseLeg:
    swingPoints: list[SwingPoint]
    direction: TrendDirection

    @property
    def startIndex(self) -> int:
        return self.swingPoints[0].barIndex

    @property
    def endIndex(self) -> int:
        return self.swingPoints[-1].barIndex

    @property
    def startPrice(self) -> float:
        return self.swingPoints[0].price

    @property
    def endPrice(self) -> float:
        return self.swingPoints[-1].price

    @property
    def barCount(self) -> int:
        return self.endIndex - self.startIndex

    @property
    def size(self) -> float:
        # Use the full price range covered by the run, not just the straight
        # start-to-end displacement - a run can end on a point that isn't its
        # most extreme one (e.g. ends on a swing low after having touched a
        # much higher swing high along the way), and using only the endpoints
        # would understate how big the move really was.
        prices = [p.price for p in self.swingPoints]
        return max(prices) - min(prices)


def buildTrendRuns(swings: list[SwingPoint]) -> list[list[SwingPoint]]:
    """
    Group consecutive swing points into maximal runs that preserve either
    uptrend structure (each new high > the run's last high, each new low >
    the run's last low) or downtrend structure (each new low < the run's
    last low, each new high < the run's last high).

    The run ends the moment a point violates that structure. The next run
    restarts from [that run's last point, the breaking point] - those two
    points are the natural start of whatever comes next (new trend, or a
    pullback that hasn't resolved yet).
    """
    if len(swings) < 2:
        return []

    runs: list[list[SwingPoint]] = []
    currentRun = [swings[0], swings[1]]
    direction = (
        TrendDirection.UP if swings[1].price > swings[0].price
        else TrendDirection.DOWN
    )

    for point in swings[2:]:
        lastSameType = next(
            (p for p in reversed(currentRun) if p.isHigh == point.isHigh), None
        )

        if lastSameType is None:
            currentRun.append(point)
            continue

        structureHolds = (
            point.price > lastSameType.price if direction == TrendDirection.UP
            else point.price < lastSameType.price
        )

        if structureHolds:
            currentRun.append(point)
        else:
            runs.append(currentRun)
            breakStart = currentRun[-1]
            currentRun = [breakStart, point]
            direction = (
                TrendDirection.UP if point.price > breakStart.price
                else TrendDirection.DOWN
            )

    runs.append(currentRun)
    return runs


def mergeConsecutiveSameDirectionLegs(
    legs: list[ImpulseLeg],
    klines: pd.DataFrame,
    atrPeriod: int = 14,
    maxGapAtrMultiple: float = 2.0,
) -> list[ImpulseLeg]:
    """
    Two confirmed legs in a row can end up pointing the same direction for
    two very different reasons:
      (a) a tiny in-between wiggle broke the strict HH/HL structure test,
          and that wiggle was too small to qualify as its own leg - there's
          no real trend change, just noise, and the two legs should be
          stitched back into one; or
      (b) a genuinely large move happened in the gap (e.g. a multi-month
          complex pullback) that failed the *reversal* confirmation check,
          not the size check - the trend never technically reversed, but
          there's real structure there that should stay visible as its own
          zone, not get silently swallowed.

    Only case (a) should be merged. This is decided by checking the price
    range actually covered by the gap itself: small gap -> merge; gap with
    a real move in it -> keep the legs separate.
    """
    if not legs:
        return legs

    atr = computeAtr(klines, atrPeriod)
    merged = [legs[0]]

    for leg in legs[1:]:
        previous = merged[-1]
        sameDirection = leg.direction == previous.direction
        gapIsSmall = False

        if sameDirection and leg.startIndex > previous.endIndex:
            gapSlice = klines.iloc[previous.endIndex: leg.startIndex + 1]
            gapRange = gapSlice["high"].max() - gapSlice["low"].min()
            gapAvgAtr = atr.iloc[previous.endIndex: leg.startIndex + 1].mean()
            if pd.notna(gapAvgAtr) and gapAvgAtr > 0:
                gapIsSmall = gapRange / gapAvgAtr < maxGapAtrMultiple
        elif sameDirection:
            gapIsSmall = True  # legs are already adjacent/overlapping - nothing to lose by stitching

        if sameDirection and gapIsSmall:
            combinedPoints = previous.swingPoints + leg.swingPoints
            merged[-1] = ImpulseLeg(swingPoints=combinedPoints, direction=leg.direction)
        else:
            merged.append(leg)

    return merged


def detectImpulseLegs(
    swings: list[SwingPoint],
    klines: pd.DataFrame,
    atrPeriod: int = 14,
    minRunAtrMultiple: float = 4.0,
    minReversalAtrMultiple: float = 1.0,
) -> list[ImpulseLeg]:
    """
    Keep only the trend runs whose total move is at least
    `minRunAtrMultiple * average ATR over the run` - filters out small/noisy
    runs so what's left are genuine, tradable impulse legs.

    A run that *opposes* the direction of the most recently confirmed impulse
    leg is a claim that the trend has reversed. Being structurally valid and
    big enough on its own isn't sufficient evidence for that claim - a large,
    clean-looking, multi-legged pullback can easily satisfy both. So an
    opposing run is only accepted if it decisively breaks beyond the prior
    impulse leg's own origin (by at least `minReversalAtrMultiple * ATR`).
    Until that happens, it's left unclassified - part of an ongoing pullback
    for step 3/4 to characterize, not a confirmed new impulse leg. A run that
    *continues* the established direction doesn't need this extra check.

    Consecutive confirmed legs that end up pointing the same direction (see
    `mergeConsecutiveSameDirectionLegs`) are stitched back together before
    returning, so the result reads as one leg per real trend, not one leg
    per structurally-clean fragment.
    """
    atr = computeAtr(klines, atrPeriod)
    runs = buildTrendRuns(swings)

    impulseLegs: list[ImpulseLeg] = []
    lastConfirmed: ImpulseLeg | None = None
    trendOrigin: float | None = None  # earliest origin of the current unbroken directional trend

    for run in runs:
        if len(run) < 2:
            continue

        startIndex = run[0].barIndex
        endIndex = run[-1].barIndex
        direction = (
            TrendDirection.UP if run[-1].price > run[0].price
            else TrendDirection.DOWN
        )
        candidate = ImpulseLeg(swingPoints=run, direction=direction)

        avgAtr = atr.iloc[startIndex: endIndex + 1].mean()
        if pd.isna(avgAtr) or avgAtr <= 0:
            continue

        if candidate.size / avgAtr < minRunAtrMultiple:
            continue

        if lastConfirmed is None:
            trendOrigin = candidate.startPrice
        elif direction != lastConfirmed.direction:
            # Reversal claim - measure the breach against the ORIGIN of the
            # whole unbroken trend so far, not just the most recently
            # confirmed leg's own start. Otherwise a trend that got split
            # into several same-direction legs (e.g. one small wiggle midway)
            # would let a later pullback "breach" a start point that was
            # never the trend's true base, confirming a reversal too easily.
            runPrices = [p.price for p in run]
            breach = (
                trendOrigin - min(runPrices) if direction == TrendDirection.DOWN
                else max(runPrices) - trendOrigin
            )
            if breach < minReversalAtrMultiple * avgAtr:
                continue  # doesn't decisively break the trend's origin - still a pullback candidate
            trendOrigin = candidate.startPrice  # genuine reversal confirmed - reset the origin
        # else: same-direction continuation - trendOrigin is left untouched

        impulseLegs.append(candidate)
        lastConfirmed = candidate

    return mergeConsecutiveSameDirectionLegs(impulseLegs, klines, atrPeriod=atrPeriod)


if __name__ == "__main__":
    from ..zigzag.swingPoints import detectSwingPoints

    klines = pd.read_parquet("data/MUUSDT_1d.parquet")
    swings = detectSwingPoints(klines, useAtr=True, atrPeriod=14, atrMultiple=2.0)
    legs = detectImpulseLegs(swings, klines, atrPeriod=14, minRunAtrMultiple=4.0)

    for i, leg in enumerate(legs):
        tag = "PRIMARY IMPULSE" if i == 0 else "later structural run (pullback-vs-new-trend TBD by step 3)"
        print(
            f"[{tag}] {leg.direction.value:5s}  "
            f"bar {leg.startIndex:>3d} -> {leg.endIndex:>3d}  "
            f"price {leg.startPrice:>9.2f} -> {leg.endPrice:>9.2f}  "
            f"size {leg.size:>8.2f}  swings {len(leg.swingPoints)}"
        )

    if legs:
        primary = legs[0]
        print(f"\nPost-impulse zone (pending pullback classification): "
              f"bar {primary.endIndex} -> {len(klines) - 1}")