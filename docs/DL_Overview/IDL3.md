---
layout: default
title: Optimizers and Regularizers (IDL3)
parent: DL Overview
nav_order: 4
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

[Ref: 11-785](https://www.youtube.com/watch?v=fMie0uWwzDQ&list=PLp-0K3kfddPzCnS4CqKphh-zT3aDwybDe&index=15&ab_channel=CarnegieMellonUniversityDeepLearning){: .btn .fs-3 .mb-4 .mb-md-0 }

# Improving over momentum update

Previously we saw how the derivatives change in subsequent steps (as we did in simple
momentum) and take a step considering the weighted average of the current and prior step
(actual implementation was a running average)

Now, we'll consider the way in which these derivatives change (this is called second moment)
which takes care of the variance in graident shifts.

![](/images/IDL3/momentum1.png)

The second moment can be implemented as shown below. As seen in our prior image, since we had
high variation along y and low variation along x, we will do:

![](/images/IDL3/momentum2.png)

## Commonly Used methods which use Second Moment

### RMS Prop

Here, let's do a running average like simple momentum, but do it on the second derivative
of the gradient. The gamma value is just a weighting factor between prior step's gradient (k-1)
and (1-gamma) is the weight applied to the current step's gradient:

![](/images/IDL3/rmsprop1.png)

Now, the way we will include this is our update will be to normalize the learning rate using
this second second moement:

![](/images/IDL3/rmsprop2.png)

Just for comparison, this is how the update step for simple momentum only scaled the preious
step's weight magnitude and did not touch learning rate.

![](/images/IDL2/Convergence16.png)

### ADAM (RMSprop with momentum)

![](/images/IDL3/adam1.png)

The reason first and second moments are scaled by the weighting factor is to ensure that
in the beginning of training, we don't let sigma and gamma terms to dominate 
(it'll slow us down)

![](/images/IDL3/adam3.png)

# Batch Normalization

### Problem with covarite shifts

![](/images/IDL3/covshfit1.png)

![](/images/IDL3/covshift2.png)

### Solution to covariate shifts

![](/images/IDL3/covshift3.png)

## Batch Norm Theory

- We do this covariate shifts typically at the at location of the affine sum (Wx + b)
- ![](/images/IDL3/batch_norm1.png)
- ![](/images/IDL3/batch_norm2.png)
- The above step will cause all training instances to have mean = 0 and variance = 1

Now, its nice to see data having low variance. However, the real issue arises when we try
to do backprop.

## Backprop through Batch Norm

Conventional backprop happens by taking a derivative of the divergence function as shown below:

![](/images/IDL3/batch_norm3.png)

**However, after batch norm, it gets tricky since our divergence will now depend on 
not only the mini-batch (training samples of mini-batch), but will now also depend on
the mean and variance of the entire mini-batch (since our mini-batch was scaled and shifted
according to the mean and variance)**

![](/images/IDL3/batch_norm4.png)

![](/images/IDL3/batch_norm6.png)

![](/images/IDL3/batch_norm5.png)

### Derivation

![](/images/IDL3/batchnormbackprop/Interview%20prep_230104_002510_1.jpg)
![](/images/IDL3/batchnormbackprop/Interview%20prep_230104_002510_2.jpg)
![](/images/IDL3/batchnormbackprop/Interview%20prep_230104_002510_3.jpg)
![](/images/IDL3/batchnormbackprop/Interview%20prep_230104_002510_4.jpg)
![](/images/IDL3/batchnormbackprop/Interview%20prep_230104_002510_5.jpg)
![](/images/IDL3/batchnormbackprop/Interview%20prep_230104_002510_6.jpg)
![](/images/IDL3/batchnormbackprop/Interview%20prep_230104_002510_7.jpg)
![](/images/IDL3/batchnormbackprop/Interview%20prep_230104_002510_8.jpg)
![](/images/IDL3/batchnormbackprop/Interview%20prep_230104_002510_9.jpg)


## Batch Norm in Test Time

Here also we need some estimate of variance as to where this test image belongs to.
We do so by using a running average over the training batches.

![](/images/IDL3/batch_norm7.png)