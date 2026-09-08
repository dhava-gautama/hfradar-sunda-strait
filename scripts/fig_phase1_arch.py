#!/usr/bin/env python3
"""Architecture diagrams for Phase 1 deep learning models: CNN, GRU, CNN-GRU.
Each model is saved as a separate figure for inline placement in the paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(PAPER_DIR, "figure")
DPI = 600

# Colors
C_INPUT = "#4472C4"
C_CNN = "#5B9BD5"
C_GRU = "#ED7D31"
C_FC = "#A5A5A5"
C_OUTPUT = "#70AD47"
C_ARROW = "#333333"
C_LABEL = "#222222"
C_POOL = "#7BAFD4"


def add_box(ax, x, y, w, h, text, color, fontsize=8, textcolor="white", alpha=0.92):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.04", linewidth=1.0,
                          edgecolor="white", facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=textcolor, zorder=4)


def add_arrow(ax, x1, y1, x2, y2, style="-|>"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle=style, mutation_scale=12,
                             linewidth=1.2, color=C_ARROW, zorder=2)
    ax.add_patch(arrow)


def add_dashed_box(ax, x, y, w, h, label, color):
    rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                           linewidth=1.2, edgecolor=color,
                           facecolor="none", linestyle="--", zorder=1)
    ax.add_patch(rect)
    ax.text(x, y + h/2 + 0.12, label, ha="center", va="bottom",
            fontsize=7, color=color, fontstyle="italic")


def draw_cnn(ax):
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Input
    add_box(ax, 1.0, 2.0, 1.6, 0.7, "$T{\\times}2{\\times}21{\\times}21$\nInput", C_INPUT, fontsize=7)
    add_arrow(ax, 1.85, 2.0, 2.6, 2.0)
    ax.text(2.2, 2.35, "stack\nchannels", fontsize=5.5, ha="center", color=C_LABEL, fontstyle="italic")

    # CNN blocks
    add_box(ax, 3.5, 3.0, 1.5, 0.5, "Conv2d(32)$\\times$2", C_CNN, fontsize=7)
    add_box(ax, 3.5, 2.3, 1.5, 0.5, "MaxPool+Drop", C_POOL, fontsize=7)
    add_box(ax, 3.5, 1.6, 1.5, 0.5, "Conv2d(64)$\\times$2", C_CNN, fontsize=7)
    add_box(ax, 3.5, 0.9, 1.5, 0.5, "MaxPool+Drop", C_POOL, fontsize=7)
    add_dashed_box(ax, 3.5, 1.95, 1.9, 2.7, "CNN encoder", C_CNN)

    add_arrow(ax, 3.5, 2.7, 3.5, 2.6)
    add_arrow(ax, 3.5, 2.0, 3.5, 1.9)
    add_arrow(ax, 3.5, 1.3, 3.5, 1.2)

    # Flatten + FC
    add_arrow(ax, 4.5, 1.95, 5.4, 1.95)
    ax.text(4.95, 2.25, "flatten", fontsize=5.5, ha="center", color=C_LABEL, fontstyle="italic")

    add_box(ax, 6.2, 1.95, 1.2, 0.6, "FC 512\nReLU", C_FC, fontsize=7, textcolor="#333")
    add_arrow(ax, 6.85, 1.95, 7.5, 1.95)

    # Output
    add_box(ax, 8.5, 1.95, 1.6, 0.7, "$2{\\times}21{\\times}21$\nSigmoid", C_OUTPUT, fontsize=7)


def draw_gru(ax):
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Input frames
    for i, lbl in enumerate(["$\\mathbf{X}(t{-}2)$", "$\\mathbf{X}(t{-}1)$", "$\\mathbf{X}(t)$"]):
        x = 0.8 + i * 1.2
        add_box(ax, x, 3.2, 1.0, 0.5, lbl, C_INPUT, fontsize=6.5)
        add_arrow(ax, x, 2.9, x, 2.4)
        ax.text(x, 2.6, "flatten", fontsize=5, ha="center", color=C_LABEL, fontstyle="italic")

    # GRU cells
    for i in range(3):
        x = 0.8 + i * 1.2
        add_box(ax, x, 2.0, 1.0, 0.55, "GRU\n256", C_GRU, fontsize=7)
        if i < 2:
            add_arrow(ax, x + 0.55, 2.0, x + 0.65, 2.0)

    add_dashed_box(ax, 2.0, 2.0, 3.9, 0.9, "", C_GRU)
    ax.text(2.0, 1.35, "882-dim input per step", fontsize=5.5, ha="center", color=C_LABEL, fontstyle="italic")

    # Final hidden → FC
    add_arrow(ax, 3.85, 2.0, 5.0, 2.0)
    ax.text(4.4, 2.25, "final\nhidden", fontsize=5.5, ha="center", color=C_LABEL, fontstyle="italic")

    add_box(ax, 5.8, 2.0, 1.1, 0.6, "FC\nSigmoid", C_FC, fontsize=7, textcolor="#333")
    add_arrow(ax, 6.4, 2.0, 7.1, 2.0)

    # Output
    add_box(ax, 8.0, 2.0, 1.6, 0.7, "$2{\\times}21{\\times}21$\nOutput", C_OUTPUT, fontsize=7)


def draw_cnn_gru(ax):
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Input frames
    for i, lbl in enumerate(["$\\mathbf{X}(t{-}2)$", "$\\mathbf{X}(t{-}1)$", "$\\mathbf{X}(t)$"]):
        x = 0.8 + i * 1.2
        add_box(ax, x, 3.4, 1.0, 0.45, lbl, C_INPUT, fontsize=6.5)

    # Shared CNN encoder
    add_box(ax, 2.0, 2.2, 2.8, 0.5, "Shared CNN encoder", C_CNN, fontsize=7)
    add_dashed_box(ax, 2.0, 2.2, 3.2, 0.85, "", C_CNN)

    for i in range(3):
        x = 0.8 + i * 1.2
        add_arrow(ax, x, 3.1, x, 2.55)

    ax.text(2.0, 1.55, "per-frame\nfeature vectors", fontsize=5.5, ha="center", color=C_LABEL, fontstyle="italic")

    # GRU
    add_arrow(ax, 3.7, 2.2, 4.8, 2.2)
    add_box(ax, 5.7, 2.2, 1.3, 0.7, "GRU\n256", C_GRU, fontsize=8)

    # FC → output
    add_arrow(ax, 6.4, 2.2, 7.2, 2.2)
    ax.text(6.8, 2.5, "final\nhidden", fontsize=5.5, ha="center", color=C_LABEL, fontstyle="italic")

    add_box(ax, 7.9, 2.2, 1.1, 0.6, "FC\nSigmoid", C_FC, fontsize=7, textcolor="#333")
    add_arrow(ax, 8.5, 2.2, 9.2, 2.2)

    add_box(ax, 10.2, 2.2, 1.6, 0.7, "$2{\\times}21{\\times}21$\nOutput", C_OUTPUT, fontsize=7)


os.makedirs(FIG_DIR, exist_ok=True)

# Generate each as a separate figure
for name, draw_fn in [("fig02_cnn", draw_cnn),
                       ("fig02_gru", draw_gru),
                       ("fig02_cnngru", draw_cnn_gru)]:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    draw_fn(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {name}.png")
