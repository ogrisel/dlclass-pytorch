"""Generate annotated two cats figures for three vision tasks.

The figures are built from the instances collected by
``collect_twocats_instances.py``:

- ``twocats_classification.jpg``: image classification, one label for the
  whole image;
- ``twocats_semantic.jpg``: semantic segmentation, one region per class;
- ``twocats_detection.jpg``: object detection, one labeled box per object;
- ``twocats_instance.jpg``: instance segmentation, one mask per object.

Usage::

    python plot_twocats_tasks.py

Requires ``matplotlib``, ``numpy`` and ``pillow``.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = HERE.parent / "convnets_detection" / "images" / "twocats.jpg"
DEFAULT_DATA = HERE / "twocats_instances.npz"

# Instance colors, and a palette for classes (assigned in alphabetical order).
INSTANCE_COLORS = ["#00ff00", "#1e90ff", "#ff9d00", "#ff00ff"]
CLASS_PALETTE = ["#00ff00", "#ff9d00", "#1e90ff", "#ff00ff"]
MASK_THRESHOLD = 0.5
JPEG_KWARGS = {"quality": 92}


def class_colors(label_names):
    """Map each class present in ``label_names`` to a color, alphabetically."""
    names = sorted(set(label_names))
    return {name: CLASS_PALETTE[i % len(CLASS_PALETTE)] for i, name in enumerate(names)}


def load_instances(data_path, class_names, min_score):
    """Load the saved instances and keep the requested classes.

    Parameters
    ----------
    data_path : path-like
        ``.npz`` file written by ``collect_twocats_instances.py``.
    class_names : list of str
        Names of the classes to keep, for instance ``["cat"]``.
    min_score : float
        Instances with a lower confidence score are discarded.

    Returns
    -------
    dict
        ``boxes`` ``(N, 4)``, ``label_names`` ``(N,)``, ``scores`` ``(N,)`` and
        ``masks`` ``(N, H, W)`` (probabilities in ``[0, 1]``), sorted from left
        to right so that the instance colors are stable.
    """
    with np.load(data_path) as data:
        keep = np.isin(data["label_names"], class_names) & (data["scores"] >= min_score)
        selected = {
            "boxes": data["boxes"][keep],
            "label_names": data["label_names"][keep],
            "scores": data["scores"][keep],
            "masks": data["masks"][keep].astype(np.float32) / 255,
        }
    order = np.argsort(selected["boxes"][:, 0])
    return {key: value[order] for key, value in selected.items()}


def new_canvas(image):
    """Create a figure with the exact size of the image and no decoration."""
    dpi = 100
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.axis("off")
    return fig, ax


def draw_mask(ax, mask, color, alpha=0.4, linewidth=4):
    """Draw a translucent filled mask with a solid outline."""
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask] = (*to_rgb(color), alpha)
    ax.imshow(overlay)
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth)


def draw_label(ax, x, y, text, color, fontsize=20):
    """Write a label on a colored background anchored at ``(x, y)``.

    Labels in the right part of the image are right-aligned so they stay
    inside the figure.
    """
    width = ax.get_xlim()[1]
    ax.text(
        x,
        y,
        text,
        color="black",
        fontsize=fontsize,
        fontweight="bold",
        va="bottom",
        ha="right" if x > 0.6 * width else "left",
        bbox={"facecolor": color, "edgecolor": color, "boxstyle": "square,pad=0.2"},
    )


def plot_semantic(image, instances, output_path):
    """Semantic segmentation: one region per class, each pixel has one class.

    The class score of a pixel is the best instance score of that class. Each
    pixel is then assigned to the class with the highest score, so the regions
    of different classes never overlap.
    """
    fig, ax = new_canvas(image)
    colors = class_colors(instances["label_names"])
    names = list(colors)
    class_scores = np.stack(
        [
            instances["masks"][instances["label_names"] == name].max(axis=0)
            for name in names
        ]
    )
    owner = class_scores.argmax(axis=0)
    foreground = class_scores.max(axis=0) >= MASK_THRESHOLD

    for k, name in enumerate(names):
        class_mask = foreground & (owner == k)
        if not class_mask.any():
            continue
        draw_mask(ax, class_mask, colors[name])
        # Anchor the class label on the top-most pixel of the region.
        ys, xs = np.nonzero(class_mask)
        draw_label(ax, xs[ys.argmin()], ys.min() - 4, name, colors[name])
    fig.savefig(output_path, pil_kwargs=JPEG_KWARGS)
    plt.close(fig)


def plot_detection(image, instances, output_path):
    """Object detection: one labeled bounding box per object."""
    fig, ax = new_canvas(image)
    colors = class_colors(instances["label_names"])
    for box, name, score in zip(
        instances["boxes"], instances["label_names"], instances["scores"]
    ):
        x1, y1, x2, y2 = box
        color = colors[name]
        ax.add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=4)
        )
        draw_label(ax, x1, y1 - 4, f"{name} {score:.2f}", color)
    fig.savefig(output_path, pil_kwargs=JPEG_KWARGS)
    plt.close(fig)


def plot_classification(image, output_path, label="Cat"):
    """Image classification: one label for the whole image, at its center."""
    fig, ax = new_canvas(image)
    height, width = image.shape[:2]
    ax.text(
        width / 2,
        height / 2,
        label,
        color="white",
        fontsize=96,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none", "boxstyle": "round,pad=0.3"},
    )
    fig.savefig(output_path, pil_kwargs=JPEG_KWARGS)
    plt.close(fig)


def plot_instance(image, instances, output_path):
    """Instance segmentation: one mask and color per object."""
    probabilities = instances["masks"]
    # Pixels claimed by several masks go to the most confident one.
    owner = probabilities.argmax(axis=0)
    foreground = probabilities.max(axis=0) >= MASK_THRESHOLD

    fig, ax = new_canvas(image)
    counts = {}
    for i, name in enumerate(instances["label_names"]):
        counts[name] = counts.get(name, 0) + 1
        color = INSTANCE_COLORS[i % len(INSTANCE_COLORS)]
        instance_mask = foreground & (owner == i)
        if not instance_mask.any():
            continue
        draw_mask(ax, instance_mask, color)
        ys, xs = np.nonzero(instance_mask)
        draw_label(ax, xs[ys.argmin()], ys.min() - 4, f"{name} {counts[name]}", color)
    fig.savefig(output_path, pil_kwargs=JPEG_KWARGS)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--classes", nargs="+", default=["cat", "couch"])
    parser.add_argument("--min-score", type=float, default=0.5)
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    instances = load_instances(args.data, args.classes, args.min_score)
    if len(instances["scores"]) == 0:
        raise SystemExit(f"No {args.classes} instance with score >= {args.min_score}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, plot in [
        ("twocats_classification.jpg", plot_classification),
        ("twocats_semantic.jpg", plot_semantic),
        ("twocats_detection.jpg", plot_detection),
        ("twocats_instance.jpg", plot_instance),
    ]:
        path = args.output_dir / name
        if plot is plot_classification:
            plot(image, path)
        else:
            plot(image, instances, path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
