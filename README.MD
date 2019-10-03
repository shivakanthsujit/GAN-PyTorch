# Implementation of GANs using PyTorch

The Generative Adversarial Network implemented in this project was based on the original paper [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) by Goodfellow et al.

It was trained to learn a simple polynomial function 
![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?x%5E%7B2%7D&plus;3x&plus;2)

ReLU and Leaky ReLU with Dropout were used in the generator and discriminator respectively along with having one-sided label smoothing. 

These optimisations were based on the recommendations from [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf) and [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf).