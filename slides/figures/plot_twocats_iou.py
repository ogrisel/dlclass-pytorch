"""Draw figures that illustrate the IoU metric on the two cats image.

The ground truth masks are the instances collected by
``collect_twocats_instances.py``. The predictions are simulated by applying a
small smooth deformation and a small shift to the ground truth label map, so
that the predicted regions overlap the ground truth without matching it.

- ``twocats_iou.jpg``: ground truth, prediction and overlap of the ``cat``
  class, with the IoU computed as the ratio of the intersection over the
  union;
- ``twocats_miou.jpg``: ground truth and prediction of all the classes, with
  the IoU of each class and the mean IoU.

Usage::

    python plot_twocats_iou.py

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
from plot_twocats_tasks import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_IMAGE,
    MASK_THRESHOLD,
    class_colors,
    load_instances,
)

HERE = Path(__file__).resolve().parent
JPEG_KWARGS = {"quality": 92}
BACKGROUND = "background"
BACKGROUND_COLOR = "#95a5a6"
MEAN_COLOR = "#34495e"
GROUND_TRUTH_COLOR = "#1e90ff"
PREDICTION_COLOR = "#ff9d00"
INTERSECTION_COLOR = "#2ecc71"
ERROR_COLOR = "#e74c3c"
OVERLAY_ALPHA = 0.6
TITLE_SIZE = 22


def ground_truth_labels(instances, class_names):
    """Build the label map of the ground truth, 0 is the background.

    Each pixel gets the class with the highest score, as in the semantic
    segmentation figure, so the regions of different classes do not overlap.

    Parameters
    ----------
    instances : dict
        Instances returned by ``load_instances``.
    class_names : list of str
        Names of the classes, class ``k`` gets the label ``k + 1``.

    Returns
    -------
    ndarray of shape (height, width)
        Integer label of each pixel.
    """
    class_scores = np.stack(
        [
            instances["masks"][instances["label_names"] == name].max(axis=0)
            for name in class_names
        ]
    )
    foreground = class_scores.max(axis=0) >= MASK_THRESHOLD
    return np.where(foreground, class_scores.argmax(axis=0) + 1, 0)


def deform_labels(labels, amplitude, shift, seed):
    """Simulate an imperfect prediction by deforming a label map.

    The label map is warped with a smooth random displacement field, then
    shifted. The deformation is small compared to the size of the objects.

    Parameters
    ----------
    labels : ndarray of shape (height, width)
        Ground truth label map.
    amplitude : float
        Maximum displacement of the random field, in pixels.
    shift : tuple of float
        Constant displacement ``(dx, dy)``, in pixels.
    seed : int
        Seed of the random generator.

    Returns
    -------
    ndarray of shape (height, width)
        Deformed label map.
    """
    rng = np.random.default_rng(seed)
    height, width = labels.shape

    def smooth_field():
        coarse = rng.uniform(-1, 1, size=(4, 6)).astype(np.float32)
        resized = Image.fromarray(coarse).resize((width, height), Image.Resampling.BICUBIC)
        return np.asarray(resized)

    dx = amplitude * smooth_field() + shift[0]
    dy = amplitude * smooth_field() + shift[1]
    ys, xs = np.mgrid[:height, :width]
    source_y = np.clip(np.rint(ys + dy), 0, height - 1).astype(int)
    source_x = np.clip(np.rint(xs + dx), 0, width - 1).astype(int)
    return labels[source_y, source_x]


def intersection_over_union(prediction, ground_truth):
    """Return the IoU of two boolean masks."""
    union = np.logical_or(prediction, ground_truth).sum()
    return np.logical_and(prediction, ground_truth).sum() / union


def overlay(masks_and_colors, shape):
    """Return an RGBA image where each mask is drawn with its color."""
    rgba = np.zeros((*shape, 4), dtype=np.float32)
    for mask, color in masks_and_colors:
        rgba[mask] = (*to_rgb(color), OVERLAY_ALPHA)
    return rgba


def show_panel(ax, image, masks_and_colors, title, title_color="black"):
    """Show the image with colored masks on top and a title."""
    ax.imshow(image)
    ax.imshow(overlay(masks_and_colors, image.shape[:2]))
    ax.set_title(title, fontsize=TITLE_SIZE, color=title_color, pad=8)
    ax.axis("off")


def plot_iou(image, ground_truth, prediction, label, class_name, output_path):
    """Ground truth, prediction and overlap of one class, with its IoU."""
    g = ground_truth == label
    p = prediction == label
    intersection = g & p
    errors = (g | p) & ~intersection
    iou = float(intersection_over_union(p, g))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    show_panel(axes[0], image, [(g, GROUND_TRUTH_COLOR)], "Ground truth $G$", GROUND_TRUTH_COLOR)
    show_panel(axes[1], image, [(p, PREDICTION_COLOR)], "Prediction $P$", PREDICTION_COLOR)
    show_panel(
        axes[2],
        image,
        [(errors, ERROR_COLOR), (intersection, INTERSECTION_COLOR)],
        "Intersection and union",
    )
    axes[2].text(
        0.5,
        -0.04,
        r"$P \cap G$ (green)    $P \cup G$ (green + red)",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.01,
        rf"$\mathrm{{IoU}}_{{\mathrm{{{class_name}}}}} = "
        rf"|P \cap G| \, / \, |P \cup G| = {iou:.2f}$",
        ha="center",
        va="bottom",
        fontsize=26,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.26, wspace=0.04)
    fig.savefig(output_path, pil_kwargs=JPEG_KWARGS, facecolor="white")
    plt.close(fig)
    return iou


def plot_miou(image, ground_truth, prediction, class_names, output_path):
    """Ground truth and prediction of all the classes, with IoU per class."""
    colors = class_colors(class_names)
    names = [BACKGROUND, *class_names]
    class_colors_list = [BACKGROUND_COLOR] + [colors[name] for name in class_names]
    ious = [
        float(intersection_over_union(prediction == label, ground_truth == label))
        for label in range(len(names))
    ]
    miou = float(np.mean(ious))

    def regions(labels):
        return [(labels == k + 1, colors[name]) for k, name in enumerate(class_names)]

    # The two images take the left part of the figure, the bars the right part
    # which leaves room for the class names.
    fig = plt.figure(figsize=(12, 3.5))
    top, bottom = 0.86, 0.12
    ax_truth = fig.add_axes([0.01, bottom, 0.325, top - bottom])
    ax_prediction = fig.add_axes([0.345, bottom, 0.325, top - bottom])
    show_panel(ax_truth, image, regions(ground_truth), "Ground truth")
    show_panel(ax_prediction, image, regions(prediction), "Prediction")

    ax = fig.add_axes([0.83, bottom, 0.15, top - bottom])
    labels = [*names, "mean"]
    values = [*ious, miou]
    bar_colors = [*class_colors_list, MEAN_COLOR]
    positions = np.arange(len(labels))[::-1]
    ax.barh(positions, values, color=bar_colors, height=0.65)
    for position, value in zip(positions, values):
        ax.text(value + 0.02, position, f"{value:.2f}", va="center", fontsize=19)
    ax.set_yticks(positions, labels, fontsize=19)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=17)
    ax.set_title("IoU per class", fontsize=TITLE_SIZE, pad=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(output_path, pil_kwargs=JPEG_KWARGS, facecolor="white")
    plt.close(fig)
    return dict(zip(names, ious)), miou


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--classes", nargs="+", default=["cat", "couch"])
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--amplitude", type=float, default=30.0)
    parser.add_argument("--shift", type=float, nargs=2, default=[14.0, 8.0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    instances = load_instances(args.data, args.classes, args.min_score)
    class_names = [str(name) for name in class_colors(instances["label_names"])]
    ground_truth = ground_truth_labels(instances, class_names)
    prediction = deform_labels(ground_truth, args.amplitude, args.shift, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "twocats_iou.jpg"
    iou = plot_iou(image, ground_truth, prediction, 1, class_names[0], path)
    print(f"Saved {path}: IoU of {class_names[0]} = {iou:.4f}")

    path = args.output_dir / "twocats_miou.jpg"
    ious, miou = plot_miou(image, ground_truth, prediction, class_names, path)
    print(f"Saved {path}: IoU per class = {ious}, mean IoU = {miou:.4f}")


if __name__ == "__main__":
    main()
