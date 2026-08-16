"""
Step 2 (visualization half): highlight the detected primary impulse leg on
top of the candlestick + swing point chart from step 1, so it's easy to
sanity-check the leg by eye against the swing markers.
"""

import pandas as pd

from .impulseLegs import ImpulseLeg
from ..zigzag.plotSwingpoints import buildSwingMarkerSeries, loadKlines
from ..zigzag.swingPoints import SwingPoint, detectSwingPoints

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from pathlib import Path


def plotCandlesWithImpulseLeg(klines: pd.DataFrame, swings: list[SwingPoint],
                               impulseLeg: ImpulseLeg | None,
                               title: str = "Kline with impulse leg",
                               savePath: str | None = None) -> None:
    highMarkers, lowMarkers = buildSwingMarkerSeries(klines, swings)
    zigZagLine = highMarkers.combine_first(lowMarkers).interpolate(limit_area="inside")

    addPlots = [
        mpf.make_addplot(highMarkers, type="scatter", markersize=80, marker="v", color="#D85A30"),
        mpf.make_addplot(lowMarkers, type="scatter", markersize=80, marker="^", color="#1D9E75"),
        mpf.make_addplot(zigZagLine, type="line", color="#5B6570", width=1.0, linestyle="--"),
    ]

    figureStyle = mpf.make_mpf_style(base_mpf_style="charles", rc={"font.size": 9})

    fig, axes = mpf.plot(
        klines,
        type="candle",
        style=figureStyle,
        addplot=addPlots,
        title=title,
        ylabel="Price",
        figsize=(12, 6),
        returnfig=True,
    )

    if impulseLeg is not None:
        axis = axes[0]
        impulseColor = "#1D9E75" if impulseLeg.direction.value == "up" else "#D85A30"

        axis.axvspan(impulseLeg.startIndex, impulseLeg.endIndex,
                     color=impulseColor, alpha=0.10, zorder=0)

        legPrices = [klines.index[p.barIndex] for p in impulseLeg.swingPoints]
        legY = [p.price for p in impulseLeg.swingPoints]
        legX = [p.barIndex for p in impulseLeg.swingPoints]
        axis.plot(legX, legY, color=impulseColor, linewidth=2.5, zorder=5,
                  label=f"impulse leg ({impulseLeg.direction.value})")
        axis.legend(loc="upper left", fontsize=8)

        postImpulseStart = impulseLeg.endIndex
        axis.axvspan(postImpulseStart, len(klines) - 1,
                     color="#5B6570", alpha=0.06, zorder=0)

    if savePath:
        Path(savePath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savePath, dpi=150, bbox_inches="tight")
        print(f"Saved chart to {savePath}")
    else:
        plt.show()


if __name__ == "__main__":
    from .impulseLegs import detectImpulseLegs

    for parquetFile, label in [
        ("data/MUUSDT_1d.parquet", "MUUSDT 1d"),
        ("data/SNDKUSDT_1d.parquet", "SNDKUSDT 1d"),
        ("data/BTCUSDT_3d.parquet", "BTCUSDT 3d"),
        ("data/BTCUSDT_4h.parquet", "BTCUSDT 4h"),
    ]:
        klinesIndexed = loadKlines(parquetFile)
        klinesFlat = klinesIndexed.reset_index()

        swings = detectSwingPoints(klinesFlat, useAtr=True, atrPeriod=14, atrMultiple=2.0)
        legs = detectImpulseLegs(swings, klinesFlat, atrPeriod=14, minRunAtrMultiple=4.0)
        primaryLeg = legs[0] if legs else None

        outPath = f"output/{parquetFile.split('/')[-1].replace('.parquet', '')}_impulse.png"
        plotCandlesWithImpulseLeg(
            klinesIndexed, swings, primaryLeg,
            title=f"{label} - primary impulse leg + post-impulse zone",
            savePath=outPath,
        )