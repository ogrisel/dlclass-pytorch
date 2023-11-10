# PyTorch version of my Deep Learning class

## Slides

- [Convolutional Neural Networks for Object Detection and Segmentation](https://ogrisel.github.io/dlclass-pytorch/slides/convnets_detection_segmentation)
- [Recommender Systems](https://ogrisel.github.io/dlclass-pytorch/slides/recsys)
- ... more to come

## Notebooks

Either:

```
git clone https://github.com/ogrisel/dlclass-pytorch
cd notebooks
jupyter lab
```

or via google colab:

- https://colab.research.google.com/github/ogrisel/dlclass-pytorch

You can also import individual notebooks from the
[`https://github.com/ogrisel/dlclass-pytorch`](https://github.com/ogrisel/dlclass-pytorch) repository into other notebook
services such has [Kaggle Code](https://www.kaggle.com/code) for instance.

## Credits

The original version written in collaboration with Charles Ollion with the
tensorflow / Keras library is still available at:

https://m2dsupsdlclass.github.io/lectures-labs/

Compared to the original repo, this repo has:

- a subset of the slides and notebooks, slightly updated to use PyTorch
- a new notebook about GPT models.

Contrary to the original repo, all the notebooks should download their data
files from online locations instead of expecting files in the same folder.
Similarly the solutions to the exercises are now included inline. This should
make it easier to use GPU-enabled notebook services such as Google Colab or
Kaggle Notebooks.
