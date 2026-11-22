"""Run a pretrained COCO Mask R-CNN on the two cats image and save the instances.

The output ``.npz`` file is the input of ``plot_twocats_tasks.py``. It holds
one entry per detected instance (box, class, score and soft mask).

Usage::

    python collect_twocats_instances.py

Requires ``torch``, ``torchvision`` and ``pillow``. The pretrained weights are
downloaded by torchvision on first use.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.transforms.functional import to_tensor

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGE = HERE.parent / "convnets_detection" / "images" / "twocats.jpg"
DEFAULT_OUTPUT = HERE / "twocats_instances.npz"


def collect_instances(image_path, score_threshold=0.5):
    """Run Mask R-CNN on an image and return the detected instances.

    Parameters
    ----------
    image_path : path-like
        Path to the input image.
    score_threshold : float, default=0.5
        Instances with a lower confidence score are discarded.

    Returns
    -------
    dict
        Arrays describing the ``N`` detected instances:

        - ``boxes``: float32 array of shape ``(N, 4)`` with ``(x1, y1, x2, y2)``
          pixel coordinates;
        - ``labels``: int64 array of shape ``(N,)`` with COCO category indices;
        - ``label_names``: array of shape ``(N,)`` with the category names;
        - ``scores``: float32 array of shape ``(N,)``;
        - ``masks``: uint8 array of shape ``(N, H, W)``, the soft mask
          probabilities rescaled to ``[0, 255]``;
        - ``model``: name of the model and weights used.
    """
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights, box_score_thresh=score_threshold
    ).eval()
    categories = weights.meta["categories"]

    image = Image.open(image_path).convert("RGB")
    with torch.inference_mode():
        (prediction,) = model([to_tensor(image)])

    labels = prediction["labels"].numpy()
    soft_masks = prediction["masks"][:, 0].numpy()  # (N, H, W) in [0, 1]
    return {
        "boxes": prediction["boxes"].numpy().astype(np.float32),
        "labels": labels,
        "label_names": np.array([categories[label] for label in labels]),
        "scores": prediction["scores"].numpy().astype(np.float32),
        "masks": np.round(soft_masks * 255).astype(np.uint8),
        "model": np.array("maskrcnn_resnet50_fpn_v2 / MaskRCNN_ResNet50_FPN_V2_COCO_V1"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()

    instances = collect_instances(args.image, args.score_threshold)
    np.savez_compressed(args.output, **instances)

    print(f"Saved {len(instances['scores'])} instances to {args.output}")
    for name, score, box in zip(
        instances["label_names"], instances["scores"], instances["boxes"]
    ):
        print(f"  {name:<10} score={score:.3f} box={box.round().astype(int).tolist()}")


if __name__ == "__main__":
    main()
