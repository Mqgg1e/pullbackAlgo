"""
Step 3 (visualization half): highlight pullback candidate zones distinctly
from impulse legs, annotated with the internal leg count - the number of
swing-to-swing segments inside the zone. That count is the clearest signal
for simple (1-2 segments) vs. complex (3+) available so far; overlap ratio
turned out not to discriminate well at daily/3-day bar granularity (see
pullbackZones.py docstring / conversation notes), so it's left out of the
on-chart label for now.
"""

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path

from ..legging.impulseLegs import ImpulseLeg
from ..zigzag.plotSwingpoints import buildSwingMarkerSeries, loadKlines
from ..zigzag.swingPoints import SwingPoint, detectSwingPoints
from .pullBackzones import PullbackZone, computeBarOverlapRatio


def plotCandlesWithPullbackZones(
    klines: pd.DataFrame,
    swings: list[SwingPoint],
    impulseLegs: list[ImpulseLeg],
    pullbackZones: list[PullbackZone],
    title: str = "Kline with impulse legs + pullback zones",
    savePath: str | None = None,
) -> None:
    highMarkers, lowMarkers = buildSwingMarkerSeries(klines, swings)
    zigZagLine = highMarkers.combine_first(lowMarkers).interpolate(limit_area="inside")

    addPlots = [
        mpf.make_addplot(highMarkers, type="scatter", markersize=55, marker="v", color="#D85A30"),
        mpf.make_addplot(lowMarkers, type="scatter", markersize=55, marker="^", color="#1D9E75"),
        mpf.make_addplot(zigZagLine, type="line", color="#5B6570", width=0.8, linestyle="--"),
    ]

    figureStyle = mpf.make_mpf_style(base_mpf_style="charles", rc={"font.size": 9})

    fig, axes = mpf.plot(
        klines,
        type="candle",
        style=figureStyle,
        addplot=addPlots,
        title=title,
        ylabel="Price",
        figsize=(15, 7),
        returnfig=True,
        warn_too_much_data=len(klines) + 1,
    )
    axis = axes[0]

    upLabelled, downLabelled = False, False
    for leg in impulseLegs:
        impulseColor = "#1D9E75" if leg.direction.value == "up" else "#D85A30"
        axis.axvspan(leg.startIndex, leg.endIndex, color=impulseColor, alpha=0.10, zorder=0)

        legX = [p.barIndex for p in leg.swingPoints]
        legY = [p.price for p in leg.swingPoints]
        labelKw = {}
        if leg.direction.value == "up" and not upLabelled:
            labelKw = {"label": "impulse leg (up)"}
            upLabelled = True
        elif leg.direction.value == "down" and not downLabelled:
            labelKw = {"label": "impulse leg (down)"}
            downLabelled = True
        axis.plot(legX, legY, color=impulseColor, linewidth=2.0, zorder=5, **labelKw)

    zoneLabelled = False
    yMin, yMax = axis.get_ylim()
    for zone in pullbackZones:
        isComplex = zone.internalLegCount >= 3  # 1-2 segments = simple, 3+ = complex (per the pullback framework)
        zoneColor = "#B8860B" if isComplex else "#94A3B8"

        axis.axvspan(zone.startIndex, zone.endIndex, facecolor=zoneColor, alpha=0.12,
                     edgecolor=zoneColor, linewidth=1.2, linestyle=(0, (4, 2)), zorder=1,
                     label="pullback zone" if not zoneLabelled else None)
        zoneLabelled = True

        midpoint = (zone.startIndex + zone.endIndex) / 2
        overlap = computeBarOverlapRatio(klines, zone.startIndex, zone.endIndex)
        tag = "complex?" if isComplex else "simple?"
        axis.annotate(
            f"{tag}\nlegs={zone.internalLegCount}\noverlap={overlap:.2f}",
            xy=(midpoint, yMax), xytext=(midpoint, yMax),
            ha="center", va="bottom", fontsize=7.5, color=zoneColor, fontweight="bold",
            annotation_clip=False,
        )

    axis.legend(loc="upper left", fontsize=8)

    if savePath:
        Path(savePath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savePath, dpi=150, bbox_inches="tight")
        print(f"Saved chart to {savePath}")
    else:
        plt.show()


if __name__ == "__main__":
    from ..legging.impulseLegs import detectImpulseLegs
    from .pullBackzones import extractPullbackZones

    datasets = [
        ("data/MUUSDT_1d.parquet", "MUUSDT 1d"),
        ("data/SNDKUSDT_1d.parquet", "SNDKUSDT 1d"),
        ("data/BTCUSDT_3d.parquet", "BTCUSDT 3d"),
    ]

    for parquetFile, label in datasets:
        klinesIndexed = loadKlines(parquetFile)
        klinesFlat = klinesIndexed.reset_index()

        swings = detectSwingPoints(klinesFlat, useAtr=True, atrPeriod=14, atrMultiple=2.0)
        legs = detectImpulseLegs(swings, klinesFlat, atrPeriod=14,
                                  minRunAtrMultiple=4.0, minReversalAtrMultiple=1.0)
        zones = extractPullbackZones(legs, swings, klinesFlat)

        outPath = f"output/{parquetFile.split('/')[-1].replace('.parquet', '')}_pullback.png"
        plotCandlesWithPullbackZones(
            klinesIndexed, swings, legs, zones,
            title=f"{label} - impulse legs + pullback zones",
            savePath=outPath,
        )