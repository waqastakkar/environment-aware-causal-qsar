#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

try:
    import matplotlib as mpl
except Exception:
    mpl = None

NATURE5 = ["#E69F00", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]


@dataclass(frozen=True)
class PlotStyle:
    font_family: str = "Times New Roman"
    font_title: int = 16
    font_label: int = 14
    font_tick: int = 12
    font_legend: int = 12
    bold_text: bool = True
    palette: Sequence[str] = tuple(NATURE5)


def add_plot_style_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--svg", action="store_true", default=True, help="Write SVG figures")
    parser.add_argument("--font", default="Times New Roman", help="Global font family")
    parser.add_argument("--bold_text", action="store_true", default=True, help="Bold all text")
    parser.add_argument("--palette", default="nature5", help="Color palette name or comma-separated colors")
    parser.add_argument("--font_title", type=int, default=16)
    parser.add_argument("--font_label", type=int, default=14)
    parser.add_argument("--font_tick", type=int, default=12)
    parser.add_argument("--font_legend", type=int, default=12)


def parse_palette(palette: str) -> list[str]:
    if palette == "nature5":
        return NATURE5
    parsed = [c.strip() for c in palette.split(",") if c.strip()]
    return parsed if parsed else NATURE5


def style_from_args(args: argparse.Namespace) -> PlotStyle:
    return PlotStyle(
        font_family=args.font,
        font_title=args.font_title,
        font_label=args.font_label,
        font_tick=args.font_tick,
        font_legend=args.font_legend,
        bold_text=bool(args.bold_text),
        palette=parse_palette(args.palette),
    )


def set_manuscript_style(style: PlotStyle, svg: bool = True) -> None:
    if mpl is None:
        raise RuntimeError("matplotlib is required for plotting")
    mpl.rcParams["savefig.format"] = "svg" if svg else mpl.rcParams.get("savefig.format", "png")
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = style.font_family
    mpl.rcParams["font.weight"] = "bold" if style.bold_text else "normal"
    mpl.rcParams["axes.labelweight"] = "bold" if style.bold_text else "normal"
    mpl.rcParams["axes.titleweight"] = "bold" if style.bold_text else "normal"
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(style.palette))
    mpl.rcParams["font.size"] = style.font_label
    mpl.rcParams["axes.titlesize"] = style.font_title
    mpl.rcParams["axes.labelsize"] = style.font_label
    mpl.rcParams["xtick.labelsize"] = style.font_tick
    mpl.rcParams["ytick.labelsize"] = style.font_tick
    mpl.rcParams["legend.fontsize"] = style.font_legend


def configure_matplotlib(style: PlotStyle, svg: bool = True) -> None:
    set_manuscript_style(style, svg=svg)


def style_axis(ax, style: PlotStyle, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    w = "bold" if style.bold_text else "normal"
    if title:
        ax.set_title(title, fontsize=style.font_title, fontweight=w)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=style.font_label, fontweight=w)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=style.font_label, fontweight=w)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(style.font_tick)
        tick.set_fontweight(w)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(style.font_legend)
            text.set_fontweight(w)
