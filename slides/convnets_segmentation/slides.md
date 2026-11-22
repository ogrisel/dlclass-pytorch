class: center, middle

# Convolutional Neural Networks for Semantic Segmentation

Olivier Grisel

`olivier.grisel@inria.fr`

Adapted from [Charles Ollion and Olivier Grisel](https://github.com/m2dsupsdlclass/lectures-labs)



---
# CNNs for computer Vision

.center[
          <img src="images/vision.png" style="width: 600px;" />
]

---
# Beyond Image Classification

### CNNs
- Previous lecture: image classification

--

### Limitations
- Mostly on centered images
- A single label for the whole image: no idea of where the objects are
- Not enough for many real world vision tasks

---
# Image Classification

.center[
          <img src="images/twocats_classification.jpg" style="width: 560px;" />
]

One label for the whole image: *what* is in the image?

---
# Semantic Segmentation

.center[
          <img src="images/twocats_semantic.jpg" style="width: 560px;" />
]

One class for each pixel: *which pixels* belong to each class?

--

- No notion of object: the two cats are merged in a single region

---
# Object Detection

.center[
          <img src="images/twocats_detection.jpg" style="width: 560px;" />
]

Next lecture: one box and one class for each object, *where* is each object?

---
# Instance Segmentation

.center[
          <img src="images/twocats_instance.jpg" style="width: 560px;" />
]

Next lecture: one mask and one class for each object, *which pixels* belong to each object?


---
class: middle, center

# Feature Maps and Fully Convolutional Networks

---
# Convolutionization of a Classification Network

.center[
  <img src="images/convolutionalization.png" style="width: 400px;" />
]

.footnote.small[
Long, Jonathan, et al. "Fully convolutional networks for semantic segmentation." CVPR 2015
]

- Slide the network with an input of `(224, 224)` over a larger image. Output of varying spatial size
--

- **Convolutionize**: 

   - remove global average pooling and flattening layers
   - replace `Linear(2048, 1000)` to $1 \times 1$ `Conv2d(2048, 1000, kernel_size=1)`


???
output map upscaled from fully convolutional network

---
# Fully Convolutional Network

.center[
          <img src="images/densefc.png" style="width: 500px;" />
]

.footnote.small[
Long et al. (CVPR 2015), see the previous slide
]

--

- Predict / backpropagate for every output pixel

--
- Aggregate maps from several convolutions at different scales for more robust results

---
# Segmentation with Transposed Convolutions

.center[
          <img src="images/deconv.png" style="width: 650px;" />
]

.footnote.small[
Noh, Hyeonwoo, et al. "Learning deconvolution network for semantic segmentation." ICCV 2015
]

--

- **Transposed convolutions** (sometimes called "deconvolutions", e.g. in the paper above) learn to **upsample** feature maps
- Despite the name, they do not invert a convolution: they only restore the spatial resolution

.center[
          <img src="images/conv_deconv.png" style="width: 350px;" />
]

--
- **Sharper masks** by using precise spatial information (first and last layers).

--
- **Global context 'understanding'** by extracting semantics of coarser regions (middle layers).

---
# Image Segmentation with U-Nets

.center[
          <img src="images/unet.png" style="width: 650px;" />
]

.footnote.small[
Ronneberger, Olaf et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
]

--
- **Skip connections** between corresponding convolution and transposed convolution layers

--
- Copy and concatenate the last feature map of the same resolution to the output of the matching transposed convolution layer.


---
## State-of-the-art & links

Recent models and checkpoints for segmentation can be browsed here:

.center[
https://huggingface.co/tasks/image-segmentation
]

--

Next lecture: object detection and instance segmentation
