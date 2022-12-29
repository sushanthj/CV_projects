---
layout: default
title: Classifiers (IDL2)
parent: DL Overview
nav_order: 3
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

[Ref: 11-785](https://www.youtube.com/watch?v=MPtEhEsgacM&list=PLp-0K3kfddPzCnS4CqKphh-zT3aDwybDe&index=11&t=1002s&ab_channel=CarnegieMellonUniversityDeepLearning){: .btn .fs-3 .mb-4 .mb-md-0 }

# Binary Classifiers and Cross Entropy Loss

This is usually used for classification. For a binary case, it is given by the following
formula:

![](/images/IDL2/CE_loss.png)

Notice that the equation uses the term divergence here. This is actually the background term
for 'loss' in our situation. **Divergence tells us how *off* we are from the correct solution**. \
Note. divergence is not direction dependent, it just tells how far away we are from the
desirable output.

More formally, loss = average divergence of our output w.r.t the ground truth

Therefore, in a binary setting, if we use softmax as our activation function -> we get the
class probablity score as the output. In the binary case we get (y, 1-y) as our output.

- let y = output of softmax for each class (we have 2 classes)
- let d = ground truth
- Now, plugging in (y, 1-y) into the Cross Entropy loss formula above
- We see that when y = 0 and when y = 1, since log(0) = -infinity and log(1) = 0
- **Therefore, if d=0 and y=1, we get infinity, if d=1 and y=0 we also get infinity**

Now it's also interesting to observe the derivative of cross entropy loss function. The 
derivative is shown below:

![](/images/IDL2/CE_loss_deriv.png)

Now, notice the following cases:
- Case 1: When d=1 and y=1, plugging into the above formula, we get ```derivative(CE_loss) = -1```
- Case 2: When d=0 and y=0, plugging into above forumula we get ```derivative(CE_loss) = 1```
- **Note, if you assumed that if output(y) = desired(d) would have zero gradient, you're wrong!**
- The above two cases are plotted below
- |     Case 1                             |      Case 1 and Case 2               |
  |:---------------------------------------|:-------------------------------------|
  | ![](/images/IDL2/CE_deriv1.png)        | ![](/images/IDL2/CE_deriv2.png)      |

However, instead of cross entropy loss, if we were to use a simple L2 error 
(sum of sqaured diffs (quadratic function)) we would get a bowl shaped instead like:

![](/images/IDL2/L2_deriv.png)

## Why is Cross Entropy better than L2?

Ans. The L2 is a quadratic loss function, which is smooth bowl. Now from the above picture, \
you can see that doing gradient descent on L2 would take so much longer than using it on the
steeper curve of the cross entropy loss!

# Multi-Class Cross Entropy

![](/images/IDL2/multi_class_CE.png)

Here we only have y_i (i.e. one class which we're looking for in our loss function)
Therefore, the derivative will look different as seen above

The problem with the above definition of CE (cross entropy) is that derivative of loss
for all other classes is zero. Which isn't desirable for fast convergence. Therefore, we
slightly modify the labels 'd' as shown below:

## Label Smoothening

Here we change our target label to (1-(K-1)*e) instead of just being 1

![](/images/IDL2/Label_smoothening.png)

## Simple 2 layer network (beautiful diagram!)

![](/images/IDL2/2layernet.png)