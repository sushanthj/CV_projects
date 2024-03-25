---
layout: page
title: Rebuilding ConvNext
permalink: /ConvNext/
nav_order: 7
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>


# ConvNext

This paper is best described in its abstract as *'We gradually “modernize” a standard ResNet
toward the design of a vision Transformer, and discover several key components that contribute
to the performance difference along the way'.*

What we'll try to learn through building ConvNext is the meaning behind these design choices
in terms of:
- Activation Functions
- Architechture
- Inductive Biases
- And more so ..

# Why I Love ConvNext

This is one of the few architechtures which I have re-written from scratch including the
Dataloaders. I used ConvNext for Face Classification and beat 250+ students and TAs in my
class on a [Kaggle Competition](https://www.kaggle.com/competitions/11-785-f23-hw2p2-classification/leaderboard).

I not only re-wrote it in a simple manner, **I also had to make many design decisions in
reducing the channel widths and reducing network depth to brind down the trainable params
from 29 Million to just 11 million**

![](/images/ConvNext/kaggle.png)

# Introduction

## Drawbacks of Vanilla Vision Transformers (ViTs)

- ViTs became famous due to their ability to scale
- With huge datasets they outperformed ResNets on Image Classification
- However, ironically, the cost of global attention (to all tokens i.e. all image patches
  fed to the transformer) grows quadratically with image size
- For real world images, this issue is a big problem!

## Enter Hierarchial ViTs like SWIN Transformer

![](/images/ConvNext/swin.png)

- Instead of just global attention, introduce attention locally to a window (red boundary)
- A fixed number of image patches form a window
- This reduces the time complexity from being quadratic in image size for generic ViTs
  to now being linear w.r.t image size

- **This linear time complexity w.r.t image size made ViTs tractable for all vision tasks like
  detection, segmentation and classification**


# Approach

The use of shifted-windows as in Swin Transformers and the learnings from the era of ViTs
motivate the authors of ConvNext to begin 'modernizing' CNNs.

They begin by taking a simple ResNet-50 model and reshaping it from the learning of ViTs. They
do this in two steps:

- New Training Methods
- New Network Architectures which include:
  - Macro Design changes
  - ResNextify
  - Inverted Bottleneck
  - Larger Kernel Sizes
  - Layer-wise micro designs

## Training Optimizations

This mainly included new optimizers, larger training epochs, and new augmentation methods. 
Specifically:

- AdamW over Adam
- Augmentations such as: Mixup, Cutmix, RandAugment, Random Erasing
- Regularization schemes including Stochastic Depth and Label Smoothing

Stochastic depth is when we choose to keep a residual block active or inactive based on
some probability (maybe bernoulli or a uniform probability distribution) as shown below:

![](/images/ConvNext/stochastic_depth.png)

## Network Modernization

### Understanding ResNets

It's beneficial to first see how a resnet works. Firstly note that we have two variants
of the ResNet block
- Simple Block (used in ResNet34)
- **BottleNeck Block** (used in all other ResNets)

![](/images/ConvNext/resnet_types.png)


The overall architechture of Resnet is captured in the below diagrams:

![](/images/ConvNext/resnet_arch.png)

Where the final 1000x1 vector is for the 1000 ImageNet classes. Also note the number of
repeating ResNet block in each layer (50-layer or ResNet50 being referred below):
- Conv2_x has 3 blocks
- Conv3_x has 4 blocks

Overall we have (3,4,6,3) as **'stage compute ratio'** as defined by authors.

### Macro Design

We saw (3,4,6,3) as **'stage compute ratio'** in ResNet50 as explained previously. In Swin-Transformer the same block distribution was (1,1,9,1).

Hence, **ConvNext tries to follow the same and uses (3,3,9,3) as the block distributions**.

### Making it more Lean (to reduce params)

However, I had to cut down on this to reduce parameter limit and changed the ratios to 
**(6,5,4,4)**. This was chosen after a few ablations but also higher numbers for the initial
blocks were chosen to allow for an optimization on the number of channels at input/output of
each ConvNext stage. Specifically:

```python
# number of channels at input/output of each res_blocks
# Updated Config
self.channel_list = [50, 175, 250, 400]

# Original Config
# self.channel_list = [96, 192, 384, 768]

# number of repeats for each res_block
# Updated Config
self.block_repeat_list = [6,5,4,4]

# Original Config
# self.block_repeat_list = [3,3,9,3]
```

As you can see, to maintain the relative number of channels at each stage (at least keep it
monotonically increasing as in the original config), I had to increase the initial block_repeats
where the channel size is small and decrease the block_repeats when channel size was larger

### ResNextify

ResNext utilized group convolution in the 3x3 conv layer of bottleneck blocks. What is group
convolution?

![](/images/ConvNext/group_conv.png)

The authors of ConvNext decided to use a special case of group convolution where the number of
groups equals number of channels. **That is literally just Depthwise Seperable Convs!!!**

#### Why Depthwise Convolutions

The simple answer is the computational complexitites:
- Depth-wise Separable = ```O(n**2*d + n*d**2)``` -> as per Attention is all you need
- Generic Convolution = ```O(n**2 * d**2)``` -> Think of n = filter size spatial, d = filter size depth (num channels)
- [Reference : *MobileNet*](https://arxiv.org/pdf/1704.04861.pdf)
- [Reference : *Attention is All You Need*](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

However, while those numbers may seem weird, for a more practical example you can
view [this post](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728#:~:text=two%20smaller%20kernels.-,Depthwise%20Separable%20Convolutions,it%20is%20more%20commonly%20used.).

**Bottomline, MobileNet shows that Depthwise Seperable Conv has much lesser FLOPs than conventional
convolution layers.**




# Appendix

## Time Complexities Analyses

### Simple Matrix Multiplication

In general if we are multiplying two matrices A (of size {N,D}) and B (of size {D,D}) then
```A@B``` will involve three nested loops, specifically:

- For each of the **N rows** in A
  - We perform **D dot products**
    - Which each involves **D multiplictions**

Hence, overall time complexity = ```N * D * D``` = ```N * D**2```

### Time Complexity Analysis in Tranformers

The transformers are seq2seq models with desired output (during training) is just the
right shifted inputs. For example if input is 'I am superman' and we are building a word2word
prediciton language model given input ```I``` the desired output is ```am``` and that makes:

- OurInput = ```<SOS> I am Superman```
- Desired output = ```I am Superman <EOS>```

Consider we have ```N``` words which we project in embedding layer where each word
gets projected to a vector of shape ```D```, then a sentence of N words will get
projected to a shape of ```N x D``` (just a matrix where num_rows = num_words and num_cols = projection_size)

Then self attention in scaled-dot-product form:

![](/images/ConvNext/scaled_dot_prod_attention.png)

Will have the following time comlexity

1. Linearly transforming the rows of ```X``` to compute the query ```Q```, key ```K```, and value ```V``` matrices, each of which has shape ```(N, D)```. This is accomplished by post-multiplying ```X``` with 3 learned matrices of shape ```(D, D)```, amounting to a computational complexity of ```O(N D^2)```.
2. Computing the layer output, specified in above equation of the paper as ```SoftMax(Q @ Kt / sqrt(d)) V```, where the softmax is computed over each row. Computing ```Q @ Kt``` has complexity ```O(N^2 D)```, and post-multiplying the resultant with ```V``` has complexity ```O(N^2 D)``` as well.

Overall the time complexity would be ```O(N^2.D + N.D^2)```

**NOTE: In the paper, they say it takes only ```O(N^2 D)``` for Self Attention, but this excludes
the calculation of Q,K,V**

#### Comparison with RNNs

RNNs have a hidden state neuron which is connected across the time series as shown below:

![](/images/ConvNext/RNN_simple.png)

The hidden neuron computation is simply: ```h(t)​ = f(U x(t)​ + W h(t−1)​)```

Hence, they are modelled as O(n * d**2) *(as it's an MLP with matrix multiplication, see Appendix)* with O(n) sequential operations

#### Comparisons with Separable and Non-Separable Convs

- Depth-wise Separable = ```O(n**2*d + n*d**2)``` = Self Attention + Feed Forward MLP
- Generic Convolution = ```O(n**2 * d**2)```

#### Conclusion:

The authors of *Attention is All You Need* therefore claim that Self Attention (```O(N**2*D)``` or truly ```O(N**2*D + N*D**2)```) is parallelizable
and faster than the next best option -> i.e. Depthwise Separable Convolution (```O(N**2*D + N*D**2)```)

Considering the true calculation of Scaled Dot Product Attention, it seems to be the same
as Depthwise Separable Convolution.
