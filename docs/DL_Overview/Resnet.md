---
layout: default
title: RESNET
parent: DL Overview
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

Convolutional Neural Nets are really good at extracting features.

We will use a RESNET in this case to extract face features and make it a face detector.

![](/images/resnet/resnet1.png)

My understanding on why we use RESNETs:
1. It prevents overfitting and limits the non-linearity to the necessary amount
   by allowing for gradients to skip neurons on the backward pass
2. It solves the issue of vanishing gradients

# Forward Prop in a RESNET

A simple network would have following form:
![](/images/resnet/resnet2.png)

Now, if we add a skip connection, we get the following structure:
![](/images/resnet/resnet3.png)

Note. The new activation is g(z + a)

# Backward Prop in a RESNET

# Why use RESNETS?

