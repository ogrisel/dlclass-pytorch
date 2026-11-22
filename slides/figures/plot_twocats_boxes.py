"""Draw figures about bounding boxes on the two cats image.

The ground truth boxes are the instances collected by
``collect_twocats_instances.py``. The predictions are simulated by shifting or
jittering the ground truth boxes.

- ``twocats_box_iou.jpg``: ground truth box, predicted box and their overlap
  for one cat, with the IoU computed as the ratio of the intersection area over
  the union area;
- ``twocats_nms.jpg``: redundant boxes around each cat before and after
  non-maximum suppression;
- ``twocats_anchors.jpg``: the anchor boxes of one location of a feature map,
  and the anchor that best matches the ground truth box of a cat.

Usage::

    python plot_twocats_boxes.py

Requires ``matplotlib``, ``numpy`` and ``pillow``.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from plot_twocats_iou import (  # noqa: E402
    ERROR_COLOR,
    GROUND_TRUTH_COLOR,
    INTERSECTION_COLOR,
    PREDICTION_COLOR,
    TITLE_SIZE,
    show_panel,
)
from plot_twocats_tasks import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_IMAGE,
    INSTANCE_COLORS,
    draw_label,
    load_instances,
    new_canvas,
)

HERE = Path(__file__).resolve().parent
JPEG_KWARGS = {"quality": 92}
STRIDE = 32
ANCHOR_SCALES = (128, 256, 512)
ANCHOR_RATIOS = (0.5, 1.0, 2.0)
ANCHOR_COLORS = {128: "#f1c40f", 256: "#00e5ff", 512: "#ff4dff"}
BEST_ANCHOR_COLOR = "#ff9d00"


def box_iou(box_a, box_b):
    """Return the IoU of two boxes given as ``(x1, y1, x2, y2)``."""
    width = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    height = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    intersection = max(width, 0) * max(height, 0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / (area_a + area_b - intersection)


def non_maximum_suppression(boxes, scores, threshold):
    """Greedy non-maximum suppression.

    Boxes are visited from the highest to the lowest score. A box is kept
    unless its IoU with an already kept box is above ``threshold``.

    Parameters
    ----------
    boxes : ndarray of shape (n, 4)
        Boxes as ``(x1, y1, x2, y2)``.
    scores : ndarray of shape (n,)
        Confidence scores.
    threshold : float
        IoU above which a box is suppressed.

    Returns
    -------
    list of int
        Indices of the kept boxes, from the highest to the lowest score.
    """
    kept = []
    for i in np.argsort(-scores):
        if all(box_iou(boxes[i], boxes[j]) <= threshold for j in kept):
            kept.append(int(i))
    return kept


def box_mask(box, shape):
    """Return the boolean mask of the pixels inside a box."""
    height, width = shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    mask = np.zeros(shape, dtype=bool)
    mask[max(y1, 0) : min(y2, height), max(x1, 0) : min(x2, width)] = True
    return mask


def draw_box(ax, box, color, linewidth=4, linestyle="-"):
    """Draw the outline of a box."""
    x1, y1, x2, y2 = box
    ax.add_patch(
        plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    )


def plot_box_iou(image, ground_truth, prediction, output_path):
    """Ground truth box, predicted box and overlap, with the IoU."""
    shape = image.shape[:2]
    g = box_mask(ground_truth, shape)
    p = box_mask(prediction, shape)
    intersection = g & p
    errors = (g | p) & ~intersection
    iou = float(box_iou(ground_truth, prediction))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    show_panel(axes[0], image, [], "Ground truth $G$", GROUND_TRUTH_COLOR)
    draw_box(axes[0], ground_truth, GROUND_TRUTH_COLOR, linewidth=5)
    show_panel(axes[1], image, [], "Prediction $P$", PREDICTION_COLOR)
    draw_box(axes[1], prediction, PREDICTION_COLOR, linewidth=5)
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
        rf"$\mathrm{{IoU}} = |P \cap G| \, / \, |P \cup G| = {iou:.2f}$",
        ha="center",
        va="bottom",
        fontsize=26,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.87, bottom=0.26, wspace=0.04)
    fig.savefig(output_path, pil_kwargs=JPEG_KWARGS, facecolor="white")
    plt.close(fig)
    return iou


def simulate_detections(boxes, scores, boxes_per_object, jitter, seed):
    """Simulate redundant detections by jittering each ground truth box.

    Parameters
    ----------
    boxes : ndarray of shape (n, 4)
        Ground truth boxes.
    scores : ndarray of shape (boxes_per_object,)
        Confidence of the detections of the first object, the detections of
        the next objects have slightly lower confidences.
    boxes_per_object : int
        Number of detections around each object.
    jitter : float
        Maximum displacement of each coordinate, in pixels.
    seed : int
        Seed of the random generator.

    Returns
    -------
    detections : ndarray of shape (n * boxes_per_object, 4)
    confidences : ndarray of shape (n * boxes_per_object,)
    """
    rng = np.random.default_rng(seed)
    detections, confidences = [], []
    for k, box in enumerate(boxes):
        for score in scores[:boxes_per_object]:
            detections.append(box + rng.uniform(-jitter, jitter, size=4))
            confidences.append(score - 0.03 * k)
    return np.array(detections), np.array(confidences)


def plot_nms(image, detections, confidences, kept, n_objects, output_path):
    """Detections before and after non-maximum suppression.

    ``detections`` holds ``len(detections) // n_objects`` consecutive boxes
    around each of the ``n_objects`` objects.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))

    show_panel(axes[0], image, [], f"Before NMS: {len(detections)} boxes")
    for box in detections:
        draw_box(axes[0], box, PREDICTION_COLOR, linewidth=3)
    # The scores of the boxes around each object are listed in a column, so
    # that they do not overlap.
    for group in np.array_split(np.arange(len(detections)), n_objects):
        x = detections[group, 0].min() + 14
        y = detections[group, 1].min() + 14
        for row, i in enumerate(group[np.argsort(-confidences[group])]):
            axes[0].text(
                x,
                y + 48 * row,
                f"{confidences[i]:.2f}",
                color="black",
                fontsize=18,
                fontweight="bold",
                va="top",
                ha="left",
                bbox={"facecolor": PREDICTION_COLOR, "edgecolor": "none", "pad": 2},
            )

    show_panel(axes[1], image, [], f"After NMS: {len(kept)} boxes")
    for rank, i in enumerate(kept):
        color = INSTANCE_COLORS[rank % len(INSTANCE_COLORS)]
        draw_box(axes[1], detections[i], color, linewidth=5)
        draw_label(axes[1], detections[i][0], detections[i][1] - 4, f"cat {confidences[i]:.2f}", color, fontsize=20)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.04)
    fig.savefig(output_path, pil_kwargs=JPEG_KWARGS, facecolor="white")
    plt.close(fig)


def make_anchors(center, scales, ratios):
    """Return the anchors centered on ``center`` and their ``(scale, ratio)``.

    An anchor of scale ``s`` and ratio ``r`` (height over width) has an area of
    ``s ** 2`` pixels, a width of ``s / sqrt(r)`` and a height of ``s * sqrt(r)``.
    """
    anchors, parameters = [], []
    for scale in scales:
        for ratio in ratios:
            width, height = scale / np.sqrt(ratio), scale * np.sqrt(ratio)
            anchors.append(
                [
                    center[0] - width / 2,
                    center[1] - height / 2,
                    center[0] + width / 2,
                    center[1] + height / 2,
                ]
            )
            parameters.append((scale, ratio))
    return np.array(anchors), parameters


def plot_anchors(image, ground_truth, center, output_path):
    """Anchors of one feature map location, and the best match of a box."""
    height, width = image.shape[:2]
    anchors, parameters = make_anchors(center, ANCHOR_SCALES, ANCHOR_RATIOS)
    ious = np.array([box_iou(anchor, ground_truth) for anchor in anchors])
    best = int(ious.argmax())

    fig, ax = new_canvas(image)
    for x in range(0, width + 1, STRIDE):
        ax.axvline(x, color="white", alpha=0.3, linewidth=1)
    for y in range(0, height + 1, STRIDE):
        ax.axhline(y, color="white", alpha=0.3, linewidth=1)
    ax.add_patch(
        plt.Rectangle(
            (center[0] - STRIDE / 2, center[1] - STRIDE / 2),
            STRIDE,
            STRIDE,
            facecolor="white",
            alpha=0.5,
        )
    )
    draw_box(ax, ground_truth, GROUND_TRUTH_COLOR, linewidth=5, linestyle="--")
    for i, ((scale, _), anchor) in enumerate(zip(parameters, anchors)):
        if i != best:
            draw_box(ax, anchor, ANCHOR_COLORS[scale], linewidth=3)
    draw_box(ax, anchors[best], BEST_ANCHOR_COLOR, linewidth=7)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    draw_label(ax, anchors[best][0], anchors[best][1] - 4, f"IoU = {ious[best]:.2f}", BEST_ANCHOR_COLOR)
    fig.savefig(output_path, pil_kwargs=JPEG_KWARGS)
    plt.close(fig)
    return float(ious[best]), parameters[best]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--shift", type=float, nargs=2, default=[80.0, 60.0])
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    instances = load_instances(args.data, ["cat"], args.min_score)
    boxes = instances["boxes"]
    if len(boxes) < 2:
        raise SystemExit(f"Expected two cats with score >= {args.min_score}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    path = args.output_dir / "twocats_box_iou.jpg"
    ground_truth = boxes[0]
    prediction = ground_truth + np.array([*args.shift, *args.shift])
    iou = plot_box_iou(image, ground_truth, prediction, path)
    print(f"Saved {path}: IoU = {iou:.4f}")

    path = args.output_dir / "twocats_nms.jpg"
    detections, confidences = simulate_detections(
        boxes[:2], scores=np.array([0.97, 0.88, 0.74, 0.61]), boxes_per_object=4, jitter=30.0, seed=args.seed
    )
    kept = non_maximum_suppression(detections, confidences, args.nms_threshold)
    plot_nms(image, detections, confidences, kept, 2, path)
    print(f"Saved {path}: kept {len(kept)} of {len(detections)} boxes")

    path = args.output_dir / "twocats_anchors.jpg"
    # Center of the feature map cell that contains the center of the first cat.
    center = (
        (ground_truth[0] + ground_truth[2]) / 2 // STRIDE * STRIDE + STRIDE / 2,
        (ground_truth[1] + ground_truth[3]) / 2 // STRIDE * STRIDE + STRIDE / 2,
    )
    iou, parameters = plot_anchors(image, ground_truth, center, path)
    print(f"Saved {path}: best anchor (scale, ratio) = {parameters}, IoU = {iou:.4f}")


if __name__ == "__main__":
    main()
