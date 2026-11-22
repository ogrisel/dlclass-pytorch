"""Draw a high level diagram of a ResNet-50 classification network.

The diagram shows the shape of the feature maps for a ``224 x 224`` input
image and highlights the five operations with a stride of 2 (the first
convolution, the max pooling and the first block of stages 2 to 4) that
downscale the spatial resolution by a factor of ``2 ** 5 = 32``.

The shapes match ``torchvision.models.resnet50``.

Usage::

    python plot_resnet_downsampling.py

Requires ``matplotlib``.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "resnet_downsampling.png"

# Sizes are in inches. The figure is small so that the text stays readable
# once it is scaled to the width of a slide.
FIGURE_SIZE = (7.4, 3.9)
COLUMN_SPACING = 0.92
MAX_HEIGHT = 2.0  # height of the feature map of the full resolution input
Y_OPERATION = 3.85
Y_STRIDE = 3.47
Y_CENTER = 2.33
Y_SHAPE = 1.27
Y_CAPTION = 0.78
Y_FACTOR = 0.35

BLOCK_COLOR = "#4c72b0"
STRIDE_COLOR = "#c0392b"
TEXT_COLOR = "#2c3e50"
MUTED_COLOR = "#7f8c8d"
FONT_SIZE = 10.5
INPUT_SIZE = 224

# (operation, stride, spatial size, channels) for each feature map.
STAGES = [
    ("Input\nimage", 1, 224, 3),
    ("7×7 conv", 2, 112, 64),
    ("max pool", 2, 56, 64),
    ("stage 1", 1, 56, 256),
    ("stage 2", 2, 28, 512),
    ("stage 3", 2, 14, 1024),
    ("stage 4", 2, 7, 2048),
]


def block_width(channels):
    """Return the drawn width of a feature map, growing with the channels."""
    return 0.12 + 0.5 * (channels / 2048) ** 0.5


def draw_feature_map(ax, x, size, channels, shade):
    """Draw a feature map as a slab, tall for a high spatial resolution."""
    height = max(MAX_HEIGHT * size / INPUT_SIZE, 0.05)
    width = block_width(channels)
    ax.add_patch(
        Rectangle(
            (x - width / 2, Y_CENTER - height / 2),
            width,
            height,
            facecolor=BLOCK_COLOR,
            alpha=shade,
            edgecolor=BLOCK_COLOR,
            linewidth=1.2,
        )
    )
    return width


def draw_arrow(ax, x_start, x_end):
    """Draw a thin arrow between two feature maps."""
    ax.annotate(
        "",
        xy=(x_end, Y_CENTER),
        xytext=(x_start, Y_CENTER),
        arrowprops={"arrowstyle": "-|>", "color": MUTED_COLOR, "linewidth": 1.2},
    )


def draw_operation(ax, x, operation, stride):
    """Write the operation and its stride above the feature map it produces."""
    ax.text(
        x,
        Y_OPERATION,
        operation,
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
        color=TEXT_COLOR,
        fontweight="bold",
    )
    if operation.startswith("Input"):
        return
    ax.text(
        x,
        Y_STRIDE,
        f"stride {stride}",
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
        color=STRIDE_COLOR if stride == 2 else MUTED_COLOR,
        fontweight="bold" if stride == 2 else "normal",
    )


def draw_factor(ax, x, factor):
    """Draw the cumulative downscaling factor in a rounded box."""
    highlighted = factor == 32
    color = STRIDE_COLOR if highlighted else TEXT_COLOR
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.28, Y_FACTOR - 0.14),
            0.56,
            0.28,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#fdecea" if highlighted else "white",
            edgecolor=color,
            linewidth=1.8 if highlighted else 1.0,
        )
    )
    ax.text(
        x,
        Y_FACTOR,
        f"÷{factor}",
        ha="center",
        va="center",
        fontsize=FONT_SIZE + (1 if highlighted else 0),
        color=color,
        fontweight="bold" if highlighted else "normal",
    )


def plot_resnet_downsampling(output_path):
    """Draw the ResNet-50 diagram and save it to ``output_path``.

    Parameters
    ----------
    output_path : path-like
        Image file to write, the format is inferred from the extension.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_xlim(0, FIGURE_SIZE[0])
    ax.set_ylim(0, FIGURE_SIZE[1])
    ax.axis("off")

    n_columns = len(STAGES) + 1
    x_first = (FIGURE_SIZE[0] - COLUMN_SPACING * (n_columns - 1)) / 2
    xs = [x_first + i * COLUMN_SPACING for i in range(n_columns)]

    factor = 1
    widths = []
    for i, (operation, stride, size, channels) in enumerate(STAGES):
        factor *= stride
        shade = 0.35 + 0.65 * i / (len(STAGES) - 1)
        widths.append(draw_feature_map(ax, xs[i], size, channels, shade))
        draw_operation(ax, xs[i], operation, stride)
        ax.text(
            xs[i],
            Y_SHAPE,
            f"{size}×{size}×{channels}",
            ha="center",
            va="top",
            fontsize=FONT_SIZE - 1,
            color=TEXT_COLOR,
        )
        draw_factor(ax, xs[i], factor)

    # Classification head: global average pooling and linear layer, the output
    # is a vector of class scores.
    x_head = xs[-1]
    ax.add_patch(
        Rectangle(
            (x_head - 0.05, Y_CENTER - 0.5),
            0.1,
            1.0,
            facecolor=BLOCK_COLOR,
            edgecolor=BLOCK_COLOR,
        )
    )
    ax.text(
        x_head,
        Y_OPERATION,
        "avg pool\n+ linear",
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
        color=TEXT_COLOR,
        fontweight="bold",
    )
    ax.text(
        x_head,
        Y_SHAPE,
        "1000\nclasses",
        ha="center",
        va="top",
        fontsize=FONT_SIZE - 1,
        color=TEXT_COLOR,
    )
    widths.append(0.1)

    for i in range(n_columns - 1):
        draw_arrow(
            ax, xs[i] + widths[i] / 2 + 0.05, xs[i + 1] - widths[i + 1] / 2 - 0.05
        )

    ax.text(
        FIGURE_SIZE[0] / 2,
        Y_CAPTION,
        "cumulative downscaling of the resolution: 5 × stride 2 "
        r"$\Rightarrow 2^5 = 32$",
        ha="center",
        va="center",
        fontsize=FONT_SIZE + 0.5,
        color=STRIDE_COLOR,
        fontstyle="italic",
    )

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output_path, dpi=250, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_resnet_downsampling(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
