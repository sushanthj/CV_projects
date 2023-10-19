---
layout: default
title: Lessons Learnt CNNs
parent: Deep Learning
nav_order: 6
---

<details open markdown="block">
  <summary>
    Table of contents
  {: .text-delta }
  </summary>
1. TOC
{:toc}
</details>

## Before you Begin

[Ref: 11-785](https://mediaservices.cmu.edu/media/Deep+Learning+%28Fall+2021%29+HW1P1P2_Bootcamp+2/1_m6nkdrxy){: .btn .fs-3 .mb-4 .mb-md-0 }

This course has become quite intense to try and document. I'll be posting highlights here
which are my biggest takeaways

# CNN Architechtures Studied

## RESNET

Convolutional Neural Nets are really good at extracting features.

![](/images/resnet/resnet1.png)

My understanding on why we use RESNETs:
1. It prevents overfitting and limits the non-linearity to the necessary amount
   by allowing for gradients to skip neurons on the backward pass
2. It solves the issue of vanishing gradients

A simple network would have following form:
![](/images/resnet/resnet2.png)

Now, if we add a skip connection, we get the following structure:
![](/images/resnet/resnet3.png)

Note. The new activation is g(z + a)

## ConvNext : A ConvNet for the 2020s

# Numbers and Math

## Shared Parameter Network (Thinking wise: Neurons -> Filters)

We previously saw that the number of params of a CNN were much lesser than an MLP. A better
understand of why this would be true can be understood from the below diagram

![](/images/IDL5/scanning_mlp_to_cnn.png)

- The above shows an MLP scanning 1D data. now, the MLP in the raw form above can be seen
taking **8 inputs at time**.
- Additionally, each layer of the MLP is seen to have neurons which do the same work. Therefore,
  such neurons can be set to hold the same **shared parameters (same color)**

If we think of only the most minimalist structure of the MLP which is required (discarding
any duplicates nuerons), we can make it a scanning MLP of the below structure:

![](/images/IDL5/mlp_to_cnn.jpg)

From the above image the equivalent CNN will have the following structure:

**The first hidden layer has 4 filters of kernel-width 2 and stride 2; the second layer has 3 filters of
kernel-width 2 and stride 2; the third layer has 2 filters of kernel-width 2 and stride 2**

Also, such a CNN which is moving over 1D input (mostly time) is called a Time Delay Neural
Network **(TDNN)**

### Implications

- Lower nubmer of params
- Due to shared params, if 4 filters in the lower layer feeds 1 filter above, then the
  gradient at the higher filter will be equally rerouted (not split!) amongst the 4 lower layer
  fiters. See the below Andrej's explanation for a backprop refresher:

  ![](/images/IDL5/backprop_refresher.png)

## Kernal, Stride, Padding and Output Shape Calculation

The pytorch website has a bad looking equation, I prefer the simple one below

```python
output_shape = (input_shape + 2*padding - kernel)/(stride) + 1
```

