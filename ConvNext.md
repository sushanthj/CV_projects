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


# Introduction

## Drawbacks of Vanilla Vision Transformers (ViTs)

- ViTs became famous due to their ability to scale
- With huge datasets they outperformed ResNets on Image Classification
- However, ironically, the cost of global attention (to all tokens i.e. all image patches
  fed to the transformer) grows quadratically with image size
- For real world images, this issue is a big problem!

### Time Complexities of Matrix Multipications (Move to appendix)

In general if we are multiplying two matrices A (of size {N,D}) and B (of size {D,D}) then
```A@B``` will involve three nested loops, specifically:

- For each of the **N rows** in A
  - We perform **D dot products**
    - Which each involves **D multiplictions**

Hence, overall time complexity = ```N * D * D``` = ```N D^2```

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




## Enter Hierarchial ViTs like SWIN Transformer

- Instead of just global attention, introduce attention locally in image patches
- This reduces the time