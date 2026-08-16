"""
Step 3 (first half): extract pullback candidate zones.

A pullback zone is simply the stretch of bars between the end of one
confirmed impulse leg and the start of the next (or, for the most recent
one, everything after the last confirmed leg up to the current bar). This
module doesn't yet classify a zone as "simple" or "complex" (that's step 4)
- it just carves the zone out and attaches a few diagnostic numbers that
make the simple/complex distinction visible: how many internal swing legs
it has, and how much consecutive bars overlap (a proxy for "boxy, sideways
chop" vs. "clean directional move").
"""

from dataclasses import dataclass

import pandas as pd

from ..legging.impulseLegs import ImpulseLeg
from ..zigzag.swingPoints import SwingPoint


@dataclass
class PullbackZone:
    startIndex: int
    endIndex: int
    precedingLeg: ImpulseLeg | None  # the impulse leg this pullback is correcting, if known
    swingPoints: list[SwingPoint]    # swing points that fall inside the zone

    @property
    def barCount(self) -> int:
        return self.endIndex - self.startIndex

    @property
    def internalLegCount(self) -> int:
        """Number of swing-to-swing segments inside the zone - the classic
        'one or two segments = simple, three or more = complex' count."""
        return max(len(self.swingPoints) - 1, 0)


def computeBarOverlapRatio(klines: pd.DataFrame, startIndex: int, endIndex: int) -> float:
    """
    Average overlap between consecutive bars' high/low ranges, as a fraction
    of their average range. A binary "do they overlap at all" check is too
    lenient - almost any two consecutive bars touch somewhat. What actually
    separates a clean trend leg from a boxy, sideways pullback is *how much*
    of each bar's range gets re-covered by its neighbor:
      - trending: each bar mostly breaks past the previous one -> low overlap
      - choppy/boxy: bars mostly sit on top of each other -> high overlap
    """
    segment = klines.iloc[startIndex: endIndex + 1]
    if len(segment) < 2:
        return 0.0

    highs = segment["high"].to_numpy()
    lows = segment["low"].to_numpy()
    ranges = highs - lows

    overlapRatios = []
    for i in range(len(segment) - 1):
        overlapAmount = max(0.0, min(highs[i], highs[i + 1]) - max(lows[i], lows[i + 1]))
        avgRange = (ranges[i] + ranges[i + 1]) / 2
        if avgRange > 0:
            overlapRatios.append(overlapAmount / avgRange)

    if not overlapRatios:
        return 0.0
    return sum(overlapRatios) / len(overlapRatios)


def extractPullbackZones(
    impulseLegs: list[ImpulseLeg],
    swings: list[SwingPoint],
    klines: pd.DataFrame,
) -> list[PullbackZone]:
    """
    Build one PullbackZone for every gap between confirmed impulse legs, plus
    a final trailing zone if the data doesn't end on a confirmed leg.
    """
    zones: list[PullbackZone] = []
    lastEnd = 0
    precedingLeg: ImpulseLeg | None = None

    for leg in impulseLegs:
        if leg.startIndex > lastEnd:
            zoneSwings = [s for s in swings if lastEnd <= s.barIndex <= leg.startIndex]
            zones.append(PullbackZone(lastEnd, leg.startIndex, precedingLeg, zoneSwings))
        precedingLeg = leg
        lastEnd = leg.endIndex

    lastBarIndex = len(klines) - 1
    if lastEnd < lastBarIndex:
        zoneSwings = [s for s in swings if s.barIndex >= lastEnd]
        zones.append(PullbackZone(lastEnd, lastBarIndex, precedingLeg, zoneSwings))

    return zones


if __name__ == "__main__":
    from ..zigzag.swingPoints import detectSwingPoints
    from ..legging.impulseLegs import detectImpulseLegs

    for parquetFile in ["data/MUUSDT_1d.parquet", "data/SNDKUSDT_1d.parquet", "data/BTCUSDT_3d.parquet"]:
        klines = pd.read_parquet(parquetFile)
        swings = detectSwingPoints(klines, useAtr=True, atrPeriod=14, atrMultiple=2.0)
        legs = detectImpulseLegs(swings, klines, atrPeriod=14,
                                  minRunAtrMultiple=4.0, minReversalAtrMultiple=1.0)
        zones = extractPullbackZones(legs, swings, klines)

        print(f"=== {parquetFile} ===")
        for zone in zones:
            overlap = computeBarOverlapRatio(klines, zone.startIndex, zone.endIndex)
            startDate = klines["openTime"].iloc[zone.startIndex].date()
            endDate = klines["openTime"].iloc[zone.endIndex].date()
            print(
                f"  bar {zone.startIndex:>4d} -> {zone.endIndex:>4d}  "
                f"({startDate} -> {endDate})  "
                f"internalLegs={zone.internalLegCount:<3d} overlapRatio={overlap:.2f}"
            )
        print()