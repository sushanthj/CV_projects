---
layout: default
title: Lessons Learnt 1
parent: Deep Learning
nav_order: 7
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

# Numbers and Math

## Shared Parameter Network (Thinking wise: Neurons -> Filters)

We previously saw that the number of params of a CNN were much lesser than an MLP. A better
understand of why this would be from the below diagram

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

# Divergence and Loss Functions

## L2 Loss

![](/images/DL_Lessons/L2.png)

- The 1/2 as you can see is a scaling factor which makes derivative clean

### Intuition behind the derivative

- d = desired value / label = 1
- y = network output = 1.5
- In the above case, increasing y will increase the loss
- Hence, the derivative (think of it like slope here) will be positive (i.e rise/run is positive)
- The magnitude of this derivative will also be dependent on error (y - d)
- Hence, the derivative formula above = (y - d) and will be positive when y > d

To do gradient descent, we will therefore go in the negative direction of this derivative
by doing:

```python
new_value_of_weights = current_value_of_weights - (derivative_of_divergence * step_size)
```

## KL Divergence and CE Loss

![](/images/DL_Lessons/KL.png)

Try to think about why we have the log term in this divergence?

### Intuition on Loss Function

- In binary classification, we can either be **fully correct** (d=0 and y=0) or **fully wrong**(d=0 and y=1)
- Now, ideally we don't want to do anything when we are fully correct
- ***'don't want to do anything'*** is just english for make the penalty zero
- When we are fully wrong, we want a harsh penalty on the network and ask it to correct itself
- ***'harsh penalty'*** is english for make the penalty infinite

- Let's plug in the values and see if the above intuition is applied. *Let d = 1 and Y = 0*
- Plugging in values, ```Div = - 1 * log(0)```, and ```log(0) = - infinity```, therefore ```Div = + infinity```
- In the above line, note that because log(0) is negative infinity, that's why we multiply by -1 in the begging

### Intuition behind Derivative

- d = desired, y = actual output
- If **d = 1** and **y = 0.9**, then increasing y will decrease the loss
- As you can see above, since increase in y will make loss go lower, the slope would be negative, i.e. derivative will be negative
- The above intuition is verified from the equation ![](/images/DL_Lessons/div_case_1.png)

- Now if **d = 0** and **y = 0.9**, then increasing y will increase loss. Hence, derivative is positive


## Why KL over L2?


# Activation Functions

![](/images/DL_Lessons/kl_vs_l2.png)

## What is Inductive Bias? How does ReLU have it?

Inductive bias is the inherent, baked-in, bias that some activation functions like ReLU has
which has started becoming evident. This is what gave rise to GELU as mentioned in
the [ConvNext Paper](https://arxiv.org/pdf/2201.03545.pdf) and why it's gained prominance
in vision transformers and modern CNNs.

### When is it bad to have inductive bias?

- The inductive bias of ReLU is that it prioritizes smooth linear boundaries which is useful
when working with very high dimensional data (as it prevents overfitting I assume).
- However, if we work on low-dimensional models, which can be simple cases like trying to
  reconstruct a singular image, or even super useful networks like NERFs
- In this [RI Seminar by Prof. Simon Lucey](https://www.youtube.com/watch?v=JOVPWLBIo5Q&t=1641s&ab_channel=CMURoboticsInstitute)
  he talks about the case where a simple network with ReLU activations is used in an encoder-decoder
  like fashion to reconstruct **one image**.
    - ![](/images/DL_Lessons/inductive_bias.gif)
- The above seems to fail and the reasoning is stated to be the inductive bias of ReLU

Here's an example of what happens when we remove the inductive bias by replacing ReLU with
another activation function (Gaussians here) which are said to have higher bandwidth (i.e. can
represent higher number of frequencies (think of a neural net as giving a signal as an output))

![](/images/DL_Lessons/without_inductive_bias.png)

### When is it good to have this inductive bias?

For large networks like vision transformers which are trained on many classes, and a ton of
data which should not be overfit, in those cases the inductive bias of ReLU is okay

**Bottom Line:
Do not use ReLU in combination with any form of position encoding (like in NERFs)**

### What is the probabalistic interpretation of Sigmoid?

