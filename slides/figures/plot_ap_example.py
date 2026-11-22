"""Draw a toy example of the Average Precision of one class.

The detections of one class are sorted by decreasing confidence. Going down the
list, each detection is either a true positive (it matches a ground truth
object with a sufficient IoU) or a false positive, which gives one point of the
precision-recall curve. The Average Precision is the area under the
precision envelope of this curve.

Usage::

    python plot_ap_example.py

Requires ``matplotlib`` and ``numpy``.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
JPEG_KWARGS = {"quality": 92}
TRUE_POSITIVE_COLOR = "#27ae60"
FALSE_POSITIVE_COLOR = "#e74c3c"
CURVE_COLOR = "#34495e"
AREA_COLOR = "#1e90ff"
TITLE_FONT = 21
TEXT_FONT = 22
TICK_FONT = 21

# (confidence, is a true positive) of the detections of the class "cat", over a
# small evaluation set that contains N_GROUND_TRUTH cats.
DETECTIONS = [
    (0.97, True),
    (0.91, True),
    (0.84, False),
    (0.77, True),
    (0.62, False),
    (0.55, False),
]
N_GROUND_TRUTH = 3


def precision_recall(detections, n_ground_truth):
    """Return the precision and the recall after each detection.

    Parameters
    ----------
    detections : list of (float, bool)
        Confidence and true positive flag of each detection.
    n_ground_truth : int
        Number of ground truth objects of the class.

    Returns
    -------
    precision, recall : ndarray
        Values after each detection, with detections sorted by decreasing
        confidence.
    """
    flags = np.array([flag for _, flag in sorted(detections, reverse=True)])
    true_positives = np.cumsum(flags)
    precision = true_positives / np.arange(1, len(flags) + 1)
    recall = true_positives / n_ground_truth
    return precision, recall


def average_precision(precision, recall):
    """Return the area under the precision envelope and the envelope.

    The envelope at a recall level is the best precision reached at this
    recall level or at a higher one.
    """
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_steps = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(recall_steps * envelope)), envelope


def plot_ap(detections, n_ground_truth, output_path):
    """Ranked detections on the left, precision-recall curve on the right."""
    precision, recall = precision_recall(detections, n_ground_truth)
    ap, envelope = average_precision(precision, recall)
    ranked = sorted(detections, reverse=True)

    fig = plt.figure(figsize=(12, 5.0))
    ax_table = fig.add_axes([0.01, 0.03, 0.66, 0.84])
    ax_table.axis("off")
    ax_table.set_title(
        f"Ranked \"cat\" detections ({n_ground_truth} cats in the ground truth)",
        fontsize=TITLE_FONT,
        loc="left",
        pad=12,
    )
    columns = [0.0, 0.155, 0.447, 0.629, 0.876]
    headers = ["Rank", "Confidence", "Match", "Precision", "Recall"]
    n_rows = len(ranked) + 1
    for x, header in zip(columns, headers):
        ax_table.text(x, 1 - 0.5 / n_rows, header, fontsize=TEXT_FONT, fontweight="bold", va="center")
    for row, ((score, flag), p, r) in enumerate(zip(ranked, precision, recall), start=1):
        y = 1 - (row + 0.5) / n_rows
        label = "TP" if flag else "FP"
        color = TRUE_POSITIVE_COLOR if flag else FALSE_POSITIVE_COLOR
        ax_table.text(columns[0], y, f"{row}", fontsize=TEXT_FONT, va="center")
        ax_table.text(columns[1], y, f"{score:.2f}", fontsize=TEXT_FONT, va="center")
        ax_table.text(columns[2], y, label, fontsize=TEXT_FONT, fontweight="bold", color=color, va="center")
        ax_table.text(columns[3], y, f"{p:.2f}", fontsize=TEXT_FONT, va="center")
        ax_table.text(columns[4], y, f"{r:.2f}", fontsize=TEXT_FONT, va="center")

    ax = fig.add_axes([0.75, 0.2, 0.22, 0.67])
    steps_recall = np.concatenate([[0.0], recall])
    steps_envelope = np.concatenate([[envelope[0]], envelope])
    ax.fill_between(steps_recall, steps_envelope, step="pre", color=AREA_COLOR, alpha=0.35)
    ax.plot(recall, precision, "o-", color=CURVE_COLOR, linewidth=2.5, markersize=9)
    ax.text(0.5, 0.18, f"AP = {ap:.2f}", fontsize=28, ha="center", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Recall", fontsize=TEXT_FONT)
    ax.set_ylabel("Precision", fontsize=TEXT_FONT)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=TICK_FONT)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=TICK_FONT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Precision-recall curve", fontsize=TITLE_FONT, pad=12)

    fig.savefig(output_path, pil_kwargs=JPEG_KWARGS, facecolor="white")
    plt.close(fig)
    return ap


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "ap_example.jpg"
    ap = plot_ap(DETECTIONS, N_GROUND_TRUTH, path)
    print(f"Saved {path}: AP = {ap:.4f}")


if __name__ == "__main__":
    main()
