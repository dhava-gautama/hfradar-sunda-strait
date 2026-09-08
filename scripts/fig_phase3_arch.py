#!/usr/bin/env python3
"""Architecture diagrams for Phase 3 autoregressive models: ConvLSTM-ED and BiEF.
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

C_INPUT = "#4472C4"
C_LSTM = "#9B59B6"
C_LSTM_FW = "#8E44AD"
C_LSTM_BW = "#2980B9"
C_MERGE = "#E67E22"
C_DEC = "#E74C3C"
C_CONV = "#95A5A6"
C_OUTPUT = "#70AD47"
C_ARROW = "#333333"
C_LABEL = "#222222"


def add_box(ax, x, y, w, h, text, color, fontsize=7.5, textcolor="white", alpha=0.92):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.04", linewidth=1.0,
                          edgecolor="white", facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=textcolor, zorder=4)


def add_arrow(ax, x1, y1, x2, y2, style="-|>", color=None):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle=style, mutation_scale=12,
                             linewidth=1.2, color=color or C_ARROW, zorder=2)
    ax.add_patch(arrow)


def add_dashed_box(ax, x, y, w, h, label, color):
    rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                           linewidth=1.2, edgecolor=color,
                           facecolor="none", linestyle="--", zorder=1)
    ax.add_patch(rect)
    ax.text(x, y + h/2 + 0.12, label, ha="center", va="bottom",
            fontsize=7, color=color, fontstyle="italic")


def draw_convlstm_ed(ax):
    ax.set_xlim(-0.3, 14)
    ax.set_ylim(-0.3, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Encoder input frames
    enc_x = [0.8, 1.8, 2.6, 3.6]
    enc_labels = ["$\\mathbf{X}(t{-}11)$", "$\\mathbf{X}(t{-}10)$", "...", "$\\mathbf{X}(t)$"]
    for fx, fl in zip(enc_x, enc_labels):
        if fl == "...":
            ax.text(fx, 3.3, "...", ha="center", va="center", fontsize=12,
                    fontweight="bold", color=C_LABEL)
        else:
            add_box(ax, fx, 3.3, 0.8, 0.45, fl, C_INPUT, fontsize=6)

    # Encoder ConvLSTM cells
    for i, fx in enumerate([0.8, 1.8, 3.6]):
        add_box(ax, fx, 2.2, 0.9, 0.55, "ConvLSTM\n64ch", C_LSTM, fontsize=6.5)
        add_arrow(ax, fx, 2.95, fx, 2.55)
        if i < 2:
            nx = [1.8, 3.6][i]
            add_arrow(ax, fx + 0.5, 2.2, nx - 0.5, 2.2)
    ax.text(2.6, 2.2, "...", ha="center", va="center", fontsize=12,
            fontweight="bold", color=C_LABEL)
    add_dashed_box(ax, 2.2, 2.2, 3.4, 0.9, "Encoder", C_LSTM)

    # Arrow to decoder
    add_arrow(ax, 4.15, 2.2, 5.3, 2.2)
    ax.text(4.7, 2.5, "$h_T, c_T$", fontsize=7, ha="center", color=C_LABEL, fontstyle="italic")

    # Decoder ConvLSTM cells (autoregressive)
    dec_x = [6.0, 7.4, 8.6, 9.8]
    dec_labels = ["ConvLSTM\n64ch", "ConvLSTM\n64ch", "...", "ConvLSTM\n64ch"]
    for i, (fx, fl) in enumerate(zip(dec_x, dec_labels)):
        if fl == "...":
            ax.text(fx, 2.2, "...", ha="center", va="center", fontsize=12,
                    fontweight="bold", color=C_LABEL)
        else:
            add_box(ax, fx, 2.2, 0.9, 0.55, fl, C_DEC, fontsize=6.5)
            if i > 0 and i < 3:
                add_arrow(ax, dec_x[i-1] + 0.5, 2.2, fx - 0.5, 2.2)
            add_box(ax, fx, 1.1, 0.8, 0.4, "1$\\times$1 Conv", C_CONV, fontsize=6, textcolor="#333")
            add_arrow(ax, fx, 1.85, fx, 1.4)

    add_arrow(ax, dec_x[0] + 0.5, 2.2, dec_x[1] - 0.5, 2.2)
    add_arrow(ax, dec_x[2] - 0.3, 2.2, dec_x[2] + 0.3, 2.2)
    add_arrow(ax, dec_x[2] + 0.5, 2.2, dec_x[3] - 0.5, 2.2)

    add_dashed_box(ax, 7.9, 2.2, 4.4, 0.9, "Decoder (autoregressive)", C_DEC)

    # Feedback arrows (autoregressive)
    for i in [0, 1]:
        fx = dec_x[i]
        nx = dec_x[i+1]
        ax.annotate("", xy=(nx, 1.55), xytext=(fx, 0.85),
                    arrowprops=dict(arrowstyle="-|>", color="#999", lw=0.8, ls="--"),
                    zorder=1)
    ax.text(6.7, 0.55, "prev. output fed back", fontsize=5.5, ha="center",
            color="#888", fontstyle="italic")

    # Output labels
    out_labels = ["$\\hat{\\mathbf{X}}(t{+}1)$", "$\\hat{\\mathbf{X}}(t{+}2)$", "$\\hat{\\mathbf{X}}(t{+}6)$"]
    for fx, ol in zip([dec_x[0], dec_x[1], dec_x[3]], out_labels):
        add_box(ax, fx, 0.3, 0.8, 0.4, ol, C_OUTPUT, fontsize=6)
        add_arrow(ax, fx, 0.85, fx, 0.55)


def draw_bief(ax):
    ax.set_xlim(-0.3, 14)
    ax.set_ylim(-0.8, 4.7)
    ax.set_aspect("equal")
    ax.axis("off")

    # Input frames
    enc_x = [0.8, 1.8, 2.6, 3.6]
    enc_labels = ["$\\mathbf{X}(t{-}11)$", "$\\mathbf{X}(t{-}10)$", "...", "$\\mathbf{X}(t)$"]
    for fx, fl in zip(enc_x, enc_labels):
        if fl == "...":
            ax.text(fx, 3.8, "...", ha="center", va="center", fontsize=12,
                    fontweight="bold", color=C_LABEL)
        else:
            add_box(ax, fx, 3.8, 0.8, 0.45, fl, C_INPUT, fontsize=6)

    # Forward encoder
    for i, fx in enumerate([0.8, 1.8, 3.6]):
        add_box(ax, fx, 2.9, 0.85, 0.5, "Fwd\nConvLSTM", C_LSTM_FW, fontsize=6)
        add_arrow(ax, fx, 3.5, fx, 3.2)
        if i < 2:
            nx = [1.8, 3.6][i]
            add_arrow(ax, fx + 0.47, 2.9, nx - 0.47, 2.9)
    ax.text(2.6, 2.9, "...", ha="center", va="center", fontsize=10,
            fontweight="bold", color=C_LABEL)

    # Backward encoder
    for i, fx in enumerate([0.8, 1.8, 3.6]):
        add_box(ax, fx, 2.0, 0.85, 0.5, "Bwd\nConvLSTM", C_LSTM_BW, fontsize=6)
        add_arrow(ax, fx, 3.5, fx, 2.3)
        if i > 0:
            px = [0.8, 1.8, 3.6][i-1]
            add_arrow(ax, fx - 0.47, 2.0, px + 0.47, 2.0)
    ax.text(2.6, 2.0, "...", ha="center", va="center", fontsize=10,
            fontweight="bold", color=C_LABEL)

    add_dashed_box(ax, 2.2, 2.45, 3.5, 1.6, "Bidirectional encoder", C_LSTM_FW)

    # Merge
    add_arrow(ax, 4.1, 2.9, 4.9, 2.5)
    add_arrow(ax, 4.1, 2.0, 4.9, 2.4)
    add_box(ax, 5.6, 2.45, 1.0, 0.55, "Merge\n1$\\times$1\nConv", C_MERGE, fontsize=6)
    ax.text(5.6, 1.7, "128$\\to$64", fontsize=5.5, ha="center", color=C_LABEL, fontstyle="italic")

    # Arrow to decoder
    add_arrow(ax, 6.15, 2.45, 6.8, 2.45)

    # Decoder (same as ConvLSTM-ED)
    dec_x = [7.5, 8.9, 10.1, 11.3]
    dec_labels = ["ConvLSTM\n64ch", "ConvLSTM\n64ch", "...", "ConvLSTM\n64ch"]
    for i, (fx, fl) in enumerate(zip(dec_x, dec_labels)):
        if fl == "...":
            ax.text(fx, 2.45, "...", ha="center", va="center", fontsize=12,
                    fontweight="bold", color=C_LABEL)
        else:
            add_box(ax, fx, 2.45, 0.9, 0.55, fl, C_DEC, fontsize=6.5)
            add_box(ax, fx, 1.3, 0.8, 0.4, "1$\\times$1 Conv", C_CONV, fontsize=6, textcolor="#333")
            add_arrow(ax, fx, 2.1, fx, 1.55)

    add_arrow(ax, dec_x[0] + 0.5, 2.45, dec_x[1] - 0.5, 2.45)
    add_arrow(ax, dec_x[2] - 0.3, 2.45, dec_x[2] + 0.3, 2.45)
    add_arrow(ax, dec_x[2] + 0.5, 2.45, dec_x[3] - 0.5, 2.45)

    add_dashed_box(ax, 9.4, 2.45, 4.4, 0.9, "Decoder (autoregressive)", C_DEC)

    # Output labels
    out_labels = ["$\\hat{\\mathbf{X}}(t{+}1)$", "$\\hat{\\mathbf{X}}(t{+}2)$", "$\\hat{\\mathbf{X}}(t{+}6)$"]
    for fx, ol in zip([dec_x[0], dec_x[1], dec_x[3]], out_labels):
        add_box(ax, fx, 0.5, 0.8, 0.4, ol, C_OUTPUT, fontsize=6)
        add_arrow(ax, fx, 1.05, fx, 0.75)


os.makedirs(FIG_DIR, exist_ok=True)

# Generate each as a separate figure
for name, draw_fn, height in [("fig03_convlstm", draw_convlstm_ed, 3.8),
                                ("fig03_bief", draw_bief, 4.5)]:
    fig, ax = plt.subplots(figsize=(12, height))
    draw_fn(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {name}.png")
