#!/usr/bin/env python3
"""Generate Figure 1, the conceptual study schematic.

The script contains presentation logic only. It does not recompute or alter any
scientific result. The editable SVG retains text as text; the PDF embeds TrueType
fonts; the PNG is exported at 450 dpi on an opaque white background.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from PIL import Image


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT / "outputs" / "postlock_upgrade" / "v5r2_figure1_reconstruction"
STEM = "Figure1_Study_design_and_utility_estimands"

WIDTH_MM = 210.0
HEIGHT_MM = 145.0
RASTER_DPI = 450

NAVY = "#173B57"
BLUE = "#2C6E9B"
BLUE_MID = "#4F819F"
BLUE_PALE = "#EAF3F8"
BLUE_PRIMARY = "#DCEEF7"
INK = "#17252F"
TEXT_MUTED = "#4E606C"
GRAY_LINE = "#7B8992"
GRAY_FILL = "#F3F5F6"
GRAY_BAND = "#F6F8F9"
WHITE = "#FFFFFF"

FONT = "Arial"


def mm_to_in(value: float) -> float:
    return value / 25.4


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str = WHITE,
    edgecolor: str = NAVY,
    linewidth: float = 1.05,
    linestyle: str = "-",
    radius: float = 2.1,
    zorder: int = 2,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.65,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def text_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    *,
    facecolor: str = WHITE,
    edgecolor: str = NAVY,
    linestyle: str = "-",
    title_color: str = INK,
    body_color: str = TEXT_MUTED,
    title_size: float = 7.2,
    body_size: float = 6.25,
    title_weight: str = "bold",
    align: str = "left",
    pad_x: float = 3.0,
    top_pad: float = 3.4,
    line_gap: float = 4.0,
):
    rounded_box(
        ax,
        x,
        y,
        w,
        h,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linestyle=linestyle,
    )
    tx = x + w / 2 if align == "center" else x + pad_x
    ha = "center" if align == "center" else "left"
    ax.text(
        tx,
        y + h - top_pad,
        title,
        ha=ha,
        va="top",
        fontsize=title_size,
        fontweight=title_weight,
        color=title_color,
        zorder=4,
    )
    title_lines = title.count("\n") + 1
    body_y = y + h - top_pad - 5.0 - (title_lines - 1) * 3.7
    for index, line in enumerate(lines):
        ax.text(
            tx,
            body_y - index * line_gap,
            line,
            ha=ha,
            va="top",
            fontsize=body_size,
            color=body_color,
            zorder=4,
        )


def estimand_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    formula: str,
    role: str | None = None,
    *,
    primary: bool = False,
):
    fill = BLUE_PRIMARY if primary else WHITE
    lw = 1.45 if primary else 1.05
    rounded_box(ax, x, y, w, h, facecolor=fill, edgecolor=BLUE if primary else NAVY, linewidth=lw)
    if primary:
        ax.plot([x + 1.0, x + 1.0], [y + 2.0, y + h - 2.0], color=BLUE, lw=2.5, zorder=4)
    ax.text(
        x + 3.2,
        y + h - 3.4,
        title,
        ha="left",
        va="top",
        fontsize=7.3,
        fontweight="bold",
        color=INK,
        zorder=4,
    )
    ax.text(
        x + 3.2,
        y + h - 9.4,
        formula,
        ha="left",
        va="top",
        fontsize=8.2,
        color=NAVY,
        zorder=4,
    )
    if role:
        ax.text(
            x + 3.2,
            y + 3.0,
            role,
            ha="left",
            va="bottom",
            fontsize=6.55,
            fontweight="bold",
            color=BLUE if primary else TEXT_MUTED,
            zorder=4,
        )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NAVY,
    linestyle: str = "-",
    linewidth: float = 1.15,
    mutation_scale: float = 8.0,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 1,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=1.2,
        shrinkB=1.2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def elbow_arrow(
    ax,
    points: list[tuple[float, float]],
    *,
    color: str,
    linestyle: str,
    linewidth: float = 1.15,
):
    vertices = points[:-1]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 1)
    path = MplPath(vertices, codes)
    ax.add_patch(
        PathPatch(
            path,
            fill=False,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            capstyle="round",
            joinstyle="round",
            zorder=1,
        )
    )
    arrow(
        ax,
        points[-2],
        points[-1],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )


def section_heading(ax, x0: float, x1: float, label: str):
    y = 131.0
    ax.text(
        (x0 + x1) / 2,
        y,
        label,
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color=BLUE,
        zorder=4,
    )
    ax.plot([x0, x1], [y - 3.1, y - 3.1], color=BLUE_MID, lw=1.0, zorder=1)


def inference_band(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    method: str,
    components: str,
    note: str | None = None,
):
    rounded_box(ax, x, y, w, h, facecolor=GRAY_BAND, edgecolor=GRAY_LINE, linewidth=1.0, radius=1.8)
    ax.plot([x + 1.0, x + 1.0], [y + 2.0, y + h - 2.0], color=BLUE, lw=2.25, zorder=4)
    ax.text(x + 4.0, y + h - 3.2, title, ha="left", va="top", fontsize=7.05, fontweight="bold", color=INK)
    ax.text(x + 4.0, y + h - 8.6, method, ha="left", va="top", fontsize=6.45, fontweight="bold", color=NAVY)
    ax.text(x + 4.0, y + h - 13.0, components, ha="left", va="top", fontsize=6.15, color=TEXT_MUTED)
    if note:
        ax.text(x + 4.0, y + 3.0, note, ha="left", va="bottom", fontsize=5.65, fontstyle="italic", color=TEXT_MUTED)


def make_figure():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT, "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "axes.unicode_minus": True,
        }
    )
    fig = plt.figure(figsize=(mm_to_in(WIDTH_MM), mm_to_in(HEIGHT_MM)), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_MM)
    ax.set_ylim(0, HEIGHT_MM)
    ax.axis("off")

    ax.text(
        105,
        140.3,
        "Study design and utility estimands",
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color=INK,
    )

    section_heading(ax, 4, 124, "TRAINING ARCHITECTURE")
    section_heading(ax, 130, 206, "PARALLEL UTILITY ESTIMANDS")

    # Source and two training branches.
    text_box(
        ax,
        4,
        84,
        28,
        30,
        "MIMIC-IV source\ntraining cohort",
        ["40,745 admissions", "First 24 h static", "tabular representation"],
        facecolor=BLUE_PALE,
        edgecolor=BLUE,
        title_size=7.0,
        body_size=6.05,
        top_pad=3.1,
        line_gap=3.8,
    )
    ax.text(18, 81.2, "Source / training", ha="center", va="top", fontsize=5.65, fontweight="bold", color=BLUE)

    text_box(
        ax,
        39,
        104,
        28,
        20,
        "Real-data training",
        ["Fixed reference models", "Logistic regression + XGBoost"],
        facecolor=WHITE,
        edgecolor=NAVY,
        title_size=7.05,
        body_size=5.85,
        line_gap=4.0,
    )
    text_box(
        ax,
        39,
        67,
        28,
        25,
        "Synthetic-data\ngeneration",
        ["3 generators × 15 seeds", "45 attempts"],
        facecolor=BLUE_PALE,
        edgecolor=BLUE,
        title_size=7.05,
        body_size=6.05,
        top_pad=3.0,
        line_gap=4.2,
    )

    arrow(ax, (32, 100.0), (39, 114.0), color=NAVY, connectionstyle="arc3,rad=-0.12")
    arrow(ax, (32, 97.0), (39, 79.5), color=BLUE, connectionstyle="arc3,rad=0.12")

    # Reliability and conditional utility are deliberately separate terminal/logical branches.
    text_box(
        ax,
        73,
        82,
        26,
        23,
        "Generation reliability",
        ["45 attempts", "3 TabDDPM single-class", "collapses"],
        facecolor=GRAY_FILL,
        edgecolor=GRAY_LINE,
        title_size=6.9,
        body_size=5.65,
        line_gap=3.55,
    )
    text_box(
        ax,
        73,
        53,
        26,
        23,
        "Conditional utility set",
        ["42 utility-estimable", "datasets"],
        facecolor=BLUE_PALE,
        edgecolor=BLUE,
        title_size=6.9,
        body_size=5.8,
        line_gap=3.65,
    )
    text_box(
        ax,
        104,
        52,
        22,
        27,
        "Synthetic-data\ntraining",
        ["One model per", "estimable dataset", "LR + XGBoost"],
        facecolor=WHITE,
        edgecolor=BLUE,
        title_size=6.7,
        body_size=5.35,
        top_pad=3.0,
        line_gap=3.45,
    )
    arrow(ax, (67, 81.5), (73, 93.0), color=GRAY_LINE, connectionstyle="arc3,rad=-0.12")
    arrow(ax, (67, 77.5), (73, 64.0), color=BLUE, connectionstyle="arc3,rad=0.12")
    arrow(ax, (99, 64.5), (104, 64.5), color=BLUE)

    # Parallel internal and external domains.
    text_box(
        ax,
        130,
        101,
        35,
        23,
        "Held-out MIMIC-IV",
        ["Internal evaluation", "Source-domain test"],
        facecolor=WHITE,
        edgecolor=NAVY,
        title_size=7.05,
        body_size=6.0,
        line_gap=4.0,
    )
    text_box(
        ax,
        171,
        101,
        35,
        23,
        "External evaluation",
        ["SICdb — primary external", "eICU-CRD — multicenter", "external evaluation"],
        facecolor=WHITE,
        edgecolor=NAVY,
        title_size=7.05,
        body_size=5.6,
        line_gap=3.8,
    )

    # IUL and EUL are parallel estimands below their corresponding domains.
    estimand_box(
        ax,
        130,
        72,
        35,
        22,
        "Internal utility loss",
        r"$\mathrm{IUL} = M_{\mathrm{RI}} - M_{\mathrm{SI}}$",
    )
    estimand_box(
        ax,
        171,
        72,
        35,
        22,
        "External utility loss",
        r"$\mathrm{EUL} = M_{\mathrm{RE}} - M_{\mathrm{SE}}$",
        "Primary estimand",
        primary=True,
    )
    estimand_box(
        ax,
        151.5,
        42,
        35,
        22,
        "Transport interaction",
        "ITL = EUL − IUL",
        "Secondary estimand",
    )

    # Each trained-model branch enters both evaluation domains directly.
    arrow(ax, (67, 114), (130, 112), color=NAVY)
    elbow_arrow(ax, [(67, 119), (168, 125), (188.5, 125), (188.5, 124)], color=NAVY, linestyle="-")
    elbow_arrow(ax, [(126, 69), (128, 69), (128, 107), (130, 107)], color=BLUE, linestyle="-")
    elbow_arrow(ax, [(126, 67), (168, 67), (168, 112), (171, 112)], color=BLUE, linestyle="-")

    # Domain-specific estimands remain parallel before symmetric convergence on ITL.
    arrow(ax, (147.5, 101), (147.5, 94), color=NAVY)
    arrow(ax, (188.5, 101), (188.5, 94), color=BLUE)
    arrow(ax, (147.5, 72), (160.0, 64), color=NAVY, connectionstyle="arc3,rad=-0.04")
    arrow(ax, (188.5, 72), (178.0, 64), color=BLUE, connectionstyle="arc3,rad=0.04")

    # Formal inference layer.
    inference_band(
        ax,
        4,
        4,
        98,
        22,
        "Generator-level inference",
        "Hierarchical bootstrap",
        "Seed stochasticity + evaluation-sample uncertainty",
    )
    inference_band(
        ax,
        108,
        4,
        98,
        22,
        "Multicenter external inference (eICU-CRD)",
        "Crossed-effects model",
        "Seed + hospital + seed×hospital + finite-sample error",
        "Hospital EUL is the primary multicenter estimand.",
    )

    return fig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [OUTPUT_DIR / f"{STEM}.{suffix}" for suffix in ("svg", "pdf", "png")]
    existing = [path for path in outputs if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing figure outputs: {joined}")

    fig = make_figure()
    fig.savefig(outputs[0], format="svg", facecolor=WHITE, transparent=False)
    fig.savefig(outputs[1], format="pdf", facecolor=WHITE, transparent=False)
    fig.savefig(outputs[2], format="png", dpi=RASTER_DPI, facecolor=WHITE, transparent=False)
    plt.close(fig)
    with Image.open(outputs[2]) as image:
        image.convert("RGB").save(outputs[2], dpi=(RASTER_DPI, RASTER_DPI), optimize=True)

    legend = (
        "**Figure 1. Study design and utility estimands.** MIMIC-IV supported parallel "
        "real-data reference training and 45 synthetic-generation attempts; all attempts "
        "contributed to generation reliability, whereas downstream utility was conditional "
        "on the 42 datasets containing both outcome classes. Real- and synthetic-trained "
        "models were evaluated in held-out MIMIC-IV, SICdb, and eICU-CRD to estimate internal "
        "utility loss (IUL) and the primary external utility loss (EUL) in parallel, with the "
        "transport interaction (ITL = EUL − IUL) treated as secondary. Generator-level "
        "hierarchical bootstrap inference incorporated seed and evaluation-sample uncertainty, "
        "and eICU-CRD multicenter inference separated seed, hospital, seed×hospital, and finite-sample error. "
        "Negative ITL indicates attenuation of the synthetic–real gap externally, not superior external performance.\n"
    )
    (OUTPUT_DIR / "Figure1_legend.md").write_text(legend, encoding="utf-8")

    provenance = {
        "figure": "Figure 1",
        "title": "Study design and utility estimands",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_sources": ["docs/ESTIMANDS.md", "configs/study_config.json"],
        "presentation_only": True,
        "dimensions_mm": {"width": WIDTH_MM, "height": HEIGHT_MM},
        "png_dpi": RASTER_DPI,
        "font": FONT,
        "formats": ["svg", "pdf", "png"],
        "files": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        },
        "journal_guidance_snapshot": {
            "journal": "npj Digital Medicine",
            "accessed": "2026-09-03",
            "url": "https://www.nature.com/npjdigitalmed/for-authors-and-referees/submission-guidelines",
            "applied": [
                "white background",
                "sans-serif lettering",
                "editable vector output",
                "RGB PNG at >=300 dpi",
                "restrained color and no decorative effects",
            ],
        },
    }
    (OUTPUT_DIR / "Figure1_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
