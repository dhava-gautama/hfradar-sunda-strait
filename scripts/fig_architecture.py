#!/usr/bin/env python3
"""Generate architecture diagram for CNN-GRU-MS (Phase 3 best model)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
C_BG = "#F5F5F5"


def add_box(ax, x, y, w, h, text, color, fontsize=9, textcolor="white", alpha=0.9):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.05", linewidth=1.2,
                          edgecolor="white", facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=textcolor, zorder=4)
    return box


def add_arrow(ax, x1, y1, x2, y2, style="-|>"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                             arrowstyle=style, mutation_scale=15,
                             linewidth=1.5, color=C_ARROW, zorder=2)
    ax.add_patch(arrow)


def add_dashed_box(ax, x, y, w, h, label, color):
    rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                           linewidth=1.5, edgecolor=color,
                           facecolor="none", linestyle="--", zorder=1)
    ax.add_patch(rect)
    ax.text(x, y + h/2 + 0.15, label, ha="center", va="bottom",
            fontsize=9, color=color, fontstyle="italic")


fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(-0.5, 14.5)
ax.set_ylim(-1.5, 4.5)
ax.set_aspect("equal")
ax.axis("off")

# ── Input frames ──────────────────────────────────────────────
frame_x = [0.8, 1.8, 2.8, 4.3]
frame_labels = ["$\\mathbf{X}(t\\!-\\!T\\!+\\!1)$", "$\\mathbf{X}(t\\!-\\!T\\!+\\!2)$",
                "...", "$\\mathbf{X}(t)$"]

for i, (fx, fl) in enumerate(zip(frame_x, frame_labels)):
    if fl == "...":
        ax.text(fx, 2.0, "...", ha="center", va="center", fontsize=16,
                fontweight="bold", color=C_LABEL)
    else:
        add_box(ax, fx, 2.0, 0.85, 0.7, fl, C_INPUT, fontsize=8)

# Input size label
ax.text(2.55, 1.15, "$T \\times 2 \\times 21 \\times 21$",
        ha="center", va="center", fontsize=8, color=C_LABEL, fontstyle="italic")

# Shared CNN label
add_dashed_box(ax, 2.55, 2.0, 4.5, 1.2, "Lookback window ($T$ frames)", C_INPUT)

# ── CNN Encoder (shared weights) ──────────────────────────────
cnn_x = 6.3
add_box(ax, cnn_x, 3.2, 1.8, 0.6, "Conv2d(32)×2", C_CNN, fontsize=8)
add_box(ax, cnn_x, 2.4, 1.8, 0.6, "MaxPool + Dropout", C_CNN, fontsize=8, alpha=0.7)
add_box(ax, cnn_x, 1.6, 1.8, 0.6, "Conv2d(64)×2", C_CNN, fontsize=8)
add_box(ax, cnn_x, 0.8, 1.8, 0.6, "MaxPool + Dropout", C_CNN, fontsize=8, alpha=0.7)

add_dashed_box(ax, cnn_x, 2.0, 2.2, 3.4, "CNN encoder (shared)", C_CNN)

# Arrows: input → CNN
add_arrow(ax, 4.75, 2.0, 5.15, 2.0)
ax.text(4.95, 2.25, "each\nframe", ha="center", va="bottom", fontsize=7,
        color=C_LABEL, fontstyle="italic")

# Arrow: CNN internal
add_arrow(ax, cnn_x, 2.85, cnn_x, 2.75)
add_arrow(ax, cnn_x, 2.05, cnn_x, 1.95)
add_arrow(ax, cnn_x, 1.25, cnn_x, 1.15)

# ── Feature vectors ──────────────────────────────────────────
ax.text(8.0, 2.25, "Feature\nvectors", ha="center", va="bottom", fontsize=7,
        color=C_LABEL, fontstyle="italic")
add_arrow(ax, 7.45, 2.0, 8.35, 2.0)

# ── GRU ───────────────────────────────────────────────────────
gru_x = 9.2
add_box(ax, gru_x, 2.0, 1.3, 1.0, "GRU\n256 units", C_GRU, fontsize=9)

# Arrow: GRU → FC
add_arrow(ax, 9.9, 2.0, 10.55, 2.0)
ax.text(10.2, 2.25, "final\nhidden", ha="center", va="bottom", fontsize=7,
        color=C_LABEL, fontstyle="italic")

# ── Fully Connected (direct multi-step) ───────────────────────
fc_x = 11.3
add_box(ax, fc_x, 2.0, 1.1, 0.9, "FC\nlayer", C_FC, fontsize=9, textcolor="#333333")

# Arrow: FC → output
add_arrow(ax, 11.9, 2.0, 12.5, 2.0)

# ── Output frames ────────────────────────────────────────────
out_x = [12.9, 13.6]
out_labels = ["$\\hat{\\mathbf{X}}(t\\!+\\!1)$", "$\\hat{\\mathbf{X}}(t\\!+\\!H)$"]

add_box(ax, out_x[0], 2.6, 0.7, 0.55, out_labels[0], C_OUTPUT, fontsize=7)
ax.text(13.25, 2.15, "...", ha="center", va="center", fontsize=12,
        fontweight="bold", color=C_LABEL, rotation=55)
add_box(ax, out_x[1], 1.5, 0.7, 0.55, out_labels[1], C_OUTPUT, fontsize=7)

add_dashed_box(ax, 13.25, 2.05, 1.3, 1.8, "$H$ output frames", C_OUTPUT)

# ── Output size label ─────────────────────────────────────────
ax.text(13.25, 0.7, "$H \\times 2 \\times 21 \\times 21$",
        ha="center", va="center", fontsize=8, color=C_LABEL, fontstyle="italic")

# ── Title / model name ────────────────────────────────────────
ax.text(7.0, 4.2, "CNN-GRU-MS: Direct multi-step architecture",
        ha="center", va="center", fontsize=12, fontweight="bold", color=C_LABEL)

# ── Key annotation ────────────────────────────────────────────
ax.text(7.0, -0.8, "Single forward pass: all $H = 6$ future frames predicted simultaneously (no autoregressive decoding)",
        ha="center", va="center", fontsize=9, color=C_LABEL, fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="#EEEEEE", ec="#CCCCCC", alpha=0.8))

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "f10.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved f10.png")
