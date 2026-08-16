"""
Step 1 (visualization half): plot a candlestick chart with detected swing
points overlaid, so the swing detection can be sanity-checked by eye.

Requires: mplfinance (pip install mplfinance)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from .swingPoints import SwingPoint, detectSwingPoints


def loadKlines(parquetPath: str) -> pd.DataFrame:
    klines = pd.read_parquet(parquetPath)
    klines = klines.set_index("openTime")
    return klines


def buildSwingMarkerSeries(klines: pd.DataFrame, swings: list[SwingPoint]) -> tuple[pd.Series, pd.Series]:
    """Build two NaN-filled series (same index as klines) with swing-high and
    swing-low prices placed at their bar, for mplfinance's addplot scatter."""
    highMarkers = pd.Series(np.nan, index=klines.index)
    lowMarkers = pd.Series(np.nan, index=klines.index)

    for swing in swings:
        barTime = klines.index[swing.barIndex]
        if swing.isHigh:
            highMarkers.loc[barTime] = swing.price
        else:
            lowMarkers.loc[barTime] = swing.price

    return highMarkers, lowMarkers


def plotCandlesWithSwings(klines: pd.DataFrame, swings: list[SwingPoint],
                           title: str = "Kline with swing points",
                           savePath: str | None = None) -> None:
    highMarkers, lowMarkers = buildSwingMarkerSeries(klines, swings)

    swingLine = pd.concat([highMarkers, lowMarkers], axis=1).min(axis=1).combine_first(
        pd.concat([highMarkers, lowMarkers], axis=1).max(axis=1)
    )
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

    if savePath:
        Path(savePath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savePath, dpi=150, bbox_inches="tight")
        print(f"Saved chart to {savePath}")
    else:
        plt.show()


if __name__ == "__main__":
    klines = loadKlines("data/SNDKUSDT_4h.parquet")
    swings = detectSwingPoints(klines.reset_index(), useAtr=True, atrPeriod=14, atrMultiple=2.0)

    plotCandlesWithSwings(
        klines, swings,
        title="SNDKUSDT_4h - swing points (ZigZag, ATR x2)",
        savePath="output/SNDKUSDT_4h_swings.png",
    )