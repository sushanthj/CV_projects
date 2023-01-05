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

# Simple 2 layer network (beautiful diagram!)

![](/images/IDL2/2layernet.png)

## Backprop

1. Derivative w.r.t to the loss was already computer in previous section.
2. Now, derivative w.r.t the activation function is shown below
   
   ![](/images/IDL2/FC_backprop3.png)
   ![](/images/IDL2/FC_backprop4.png)

3. Computing derivate w.r.t one weight (one weight connects one neuron in layer N-1 to \
   another neuron in layer 2)
   
   ![](/images/IDL2/FC_backprop1.png)

4. Computing the derivative w.r.t y (y = output of activation function) \
   Here, one neuron will have effect on all the neurons in the next layer. This is why we need
   to sum the derivates of z (z = wx + b of next layer) w.r.t y(from previous layer)
   
   ![](/images/IDL2/FC_backprop2.png)

# Special Cases

## Scalar vs Vector activations

We assumed activation to be neuron specific. However, this may not be the case!

![](/images/IDL2/special_case1.png)

![](/images/IDL2/special_case2.png)

Also, the backprop gets bit murky as well

|     Scalar Activation                  |      Vector Activation               |
|:---------------------------------------|:-------------------------------------|
| ![](/images/IDL2/special_case3.png)    | ![](/images/IDL2/special_case4.png)  |

### Example of a vector activation: softmax activation

![](/images/IDL2/special_case5.png)

## Sub-gradients

For RELU's, the origin is not smooth and the gradient cannot be computed. Instead we use a \
sub-gradient which is shown as multiple lines in the figure below. However, we just use
the sub-gradient line which is parallel to the x-axis and define the gradient as = 1 at origin

![](/images/IDL2/special_case6.png)

# Training Process

## Vector Formulation

### Forward Pass
In the below picture, the first row of weights vector represents all of the weights going to
the first neuron.

![](/images/IDL2/training1.png)
![](/images/IDL2/training2.png)

### Backward Pass

Now, if z = output of affine function (wx + b) and f(z) = output of activation \
Having vectorized activations will cause the below **backprop through y = f(**z)

![](/images/IDL2/training3.png)

The Jacobian will therefore be the multivariant form of gradient, giving us the
direction in which incresing delta(z) will cause the max increase in delta(y)

**Rule of thumb: the derivative of [y(scalar) = f(z(matrix))] = matrix.T shape** \
**Extension: the derivative of scalar(row_vector) = column vector**

#### Special Cases

![](/images/IDL2/training4.png)

![](/images/IDL2/training5.png)

## Backward pass summary

1. For the first gradient, we calculate the fastest increase in Y which gets the fastest
   increase in loss, where loss is given as
   
   ![](/images/IDL2/training6.png)
   
   The derivative therefore becomes \
   ![](/images/IDL2/training7.png)
   
   Where Y = column vector, therefore derivative of Y w.r.t Divergence(loss) = delta(Y)*Div shown in picture above. This grad(y) is a row vector! 

2. Now we compute derivative through affine variable z = wx+b, then we do
   ![](/images/IDL2/training8.png)

3. Now, derivative w.r.t previous Y (Y(n-1)) will be
   ![](/images/IDL2/training9.png)

4. **Remember, as we go back we just post multiply by:**
   - A jacobian if it's an activation layer (vector activation)
   - A weight matrix for an affine layer (scalar activation)

5. Now, two more things tbd are derivatives on weights and biases!
6. Since bias should be the same size as our vector activation (z) therefore the
   defivative w.r.t the bias is

   ![](/images/IDL2/training10.png)

7. Similarly, the derivative of the weights is given by
   ![](/images/IDL2/training11.png) which will have the same shape as the weight matrix

8. **Remember, all the shapes of the derivatives should match the shapes of the 
   weights or the bias itself**

9. ![](/images/IDL2/training12.png)

## Loss Surface

- The common hypothesis is that in large networks there are lot more saddle points
than global minima.
- A saddle point is defined as one where moving in one direction increases loss and
  moving in other direction decreases. *I.e. depending on which direciton you're looking at
  you can be at a minima or maxima.* **Also the slope at saddle points is zero (therefore, 
  you'll get stuck with gradient descent)**

![](/images/IDL2/training13.png)

# Issues with Convergence

![](/images/IDL2/convergence1.png)

![](/images/IDL2/convergence2.png)

![](/images/IDL2/convergence3.png)

## Convergence for a Convex Problem

Consider a simple quadratic case

![](/images/IDL2/convergence5.jpg)

![](/images/IDL2/convergence4.png)

**Note. Optimizing w.r.t the second order is called Newton's method**

### Multivariate Convex

![](/images/IDL2/convergence6.png)

Now, the A matrix will introduce different slopes in different axes since it's
multivariate. To mitigate this we do

![](/images/IDL2/convergence3.png)

Now, you see that the A matrix has been removed **(think of it as having become identity)**

Then as we saw in our derivation above, the optimal step size will then become \
inverse(A) = inverse(I) = 1. 

**Therefore, the optimal step size is now the same in all dimensions**

#### The math for the above steps

![](/images/IDL2/convergence7.png)
![](/images/IDL2/convergence8.png)

Points to note

1. In the simple scalar quadratic space the optimal step size was one value
2. In the multivariate space, we need to scale and then find the optimal step
3. However, after scaling we can still achieve single step move to global minima
   even in multivariate space, we just need to find the inverse of a matrix

### General Case of Convex Functions (function has higher order, i.e. not a quadratic)

- Even in such cases we can find Taylor expansions and just truncate upto second order
- In such cases it'll just be an approximation, but let's live with that
- ![](/images/IDL2/convergence9.png)
- **Here we see that the second derivative is replaced by a Hessian**
- Therefore in this case, the **optimum step size would be the inverse(Hessian)**
- The normalized and optimal update step in gradient descent form is shown below
  ![](/images/IDL2/convergence10.png)

#### Issues with the above process

![](/images/IDL2/convergence11.png)

![](/images/IDL2/convergence12.png)



#### Solutions

![](/images/IDL2/convergence13.png)

![](/images/IDL2/convergence14.png)

![](/images/IDL2/convergence15.png)

![](/images/IDL2/Convergence16.png)

In the momentum method, the first term in RHS is the
scaled term of previous weight update being addes to the current
update step. 

- Big red vector = previous update step
- Blue vector = 2nd term of RHS above
- Small red vector = scaled version of big red vector
- black vector = final update (LHS term)

# SGD vs Batch GD

## SGD

In SGD we update after every training instance. The caveat for convergence is that
the increments should be small and not too large. The increments should also shrink
so that we don't keep shifting around the decision boundary too much due to to just
one training instance.

![](/images/IDL2/sgd1.png)

If we define epsilon to be the margin by which we need to be within to have 'converged', then
using the above optimal learning rate of (1/k) where **k = no. of layers**, we see that after
one iteration we should be within (1/k)*desired_range.

![](/images/IDL2/sgd2.png)

Therefore if we only need to be epsilon*desired range, we can reach it in O(1/epsilon)

## Batch GD

![](/images/IDL2/batch_gd1.png)

## Problems with SGG

1. If our job was to minimize the shaded area in the below picture (shaded area = divergence)
   then, we would want to push the red line up or down (blue = ground truth)
2. If we look at the curve at it's current location, we would want to move the red curve down
   drastically.
   ![](/images/IDL2/sgd23.png)
3. In the below picture, we would want to push up our red curve drastically
   ![](/images/IDL2/sgd4.png)
4. Therefore, the problem becomes that the estimated loss and subsequent update has too
   high a variance!
5. However, despite all this SGD is fast since it works only on 1 sample at a time

## Middle Ground Solution: Mini Batches

Here we compute the average loss over a mini-batch and use this averaged loss for update

![](/images/IDL2/mini_batch1.png)

**But how does the variance of mini-batch compare to that of full-batch gradient descent?**

- Variance of mini-batch GD where b = batch size is:
  ![](/images/IDL2/batch_gd_1.png)

- Variace of full-batch GD will be (1/N) instead of (1/b)

- Now, if we have 1000 training samples, it can be seen that 1/100 is small enough
  that it won't make that much of a difference if it's 1/100 or 1/1000. This is why
  mini-batching works, i.e. even with 100 samples we capture almost the same variance
  as we would if we took all training samples into consideration

![](/images/IDL2/mini_batch_gd2.png)


