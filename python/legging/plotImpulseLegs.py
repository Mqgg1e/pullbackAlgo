"""
Step 2 (visualization half): highlight ALL detected impulse legs on top of
the candlestick + swing point chart from step 1. Each leg is drawn as a
thick colored line with a shaded background; the gaps BETWEEN legs (and
after the last one) are what's left undetermined - pullback candidates for
step 3/4 to characterize, not impulse legs themselves.
"""

import pandas as pd

from .impulseLegs import ImpulseLeg
from ..zigzag.plotSwingpoints import buildSwingMarkerSeries, loadKlines
from ..zigzag.swingPoints import SwingPoint, detectSwingPoints

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from pathlib import Path


def plotCandlesWithImpulseLegs(klines: pd.DataFrame, swings: list[SwingPoint],
                                impulseLegs: list[ImpulseLeg],
                                title: str = "Kline with impulse legs",
                                savePath: str | None = None) -> None:
    highMarkers, lowMarkers = buildSwingMarkerSeries(klines, swings)
    zigZagLine = highMarkers.combine_first(lowMarkers).interpolate(limit_area="inside")

    addPlots = [
        mpf.make_addplot(highMarkers, type="scatter", markersize=60, marker="v", color="#D85A30"),
        mpf.make_addplot(lowMarkers, type="scatter", markersize=60, marker="^", color="#1D9E75"),
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
        figsize=(14, 7),
        returnfig=True,
        warn_too_much_data=len(klines) + 1,
    )
    axis = axes[0]

    lastEnd = 0
    upLabelled, downLabelled, pullbackLabelled = False, False, False

    for leg in impulseLegs:
        # gap before this leg = undetermined / pullback candidate zone
        if leg.startIndex > lastEnd:
            axis.axvspan(lastEnd, leg.startIndex, color="#5B6570", alpha=0.06, zorder=0,
                         label="undetermined / pullback" if not pullbackLabelled else None)
            pullbackLabelled = True

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
        axis.plot(legX, legY, color=impulseColor, linewidth=2.2, zorder=5, **labelKw)

        lastEnd = leg.endIndex

    if lastEnd < len(klines) - 1:
        axis.axvspan(lastEnd, len(klines) - 1, color="#5B6570", alpha=0.06, zorder=0,
                     label="undetermined / pullback" if not pullbackLabelled else None)

    axis.legend(loc="upper left", fontsize=8)

    if savePath:
        Path(savePath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savePath, dpi=150, bbox_inches="tight")
        print(f"Saved chart to {savePath}")
    else:
        plt.show()


if __name__ == "__main__":
    from .impulseLegs import detectImpulseLegs

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
        print(f"{label}: {len(legs)} impulse legs detected")

        outPath = f"output/{parquetFile.split('/')[-1].replace('.parquet', '')}_impulse.png"
        plotCandlesWithImpulseLegs(
            klinesIndexed, swings, legs,
            title=f"{label} - all impulse legs + undetermined zones",
            savePath=outPath,
        )