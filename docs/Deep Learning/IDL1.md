---
layout: default
title: MLPs (IDL1)
parent: Deep Learning
nav_order: 2
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

[Ref: 11-785](https://www.youtube.com/watch?v=tO3xU2t_wTY&list=PLp-0K3kfddPzCnS4CqKphh-zT3aDwybDe&index=11&ab_channel=CarnegieMellonUniversityDeepLearning){: .btn .fs-3 .mb-4 .mb-md-0 }

# Multi-Layer Perceptrons Basics

These are machines that can model any function in the world! For now, let's start with simple functions like boolean gates and build our way up.

The basic working is shown below:

![](/images/IDL/MLP0.png)

## Perceptron as a boolean gate

![](/images/IDL/MLP1.png)

- Each perceptron seen above is a an addition gate
- The sum is computed, and the threshold value is given by the number inside the circle
- Therefore, the number dictates what type of gate it functions as

### Recap types of gates:

[Andrej Reference](https://cs231n.github.io/optimization-2/)

- Add gate
- Max gate
- Multiply gate

### XOR Gate

These gates are activated only if the inputs are (1,0) or (0,1). These are bit
tricky and need to be modelled with a network of perceptrons:

![](/images/IDL/MLP3.png)

**Therefore, it can be seen that combining MLPs in such a manner, one can 
say that MLPs are universal boolean functions**

**We can also claim that any boolean function can be modelled with just 1 hidden layer**

Reason:
![](/images/IDL/MLP4.png)


## Why do we need depth?

Let's take a slightly difficult case (say an XOR)

![](/images/IDL/MLP5.png)

![](/images/IDL/MLP6.png)

However, if we model the same with XORs depthwise, we get:

![](/images/IDL/MLP7.png)

## Perceptrons as Linear Classifiers

If we have 2 boolean inputs, we can have 4 combinations:
- (0,0)
- (0,1)
- (1,0)
- (1,1)

Now, using an **OR gate, NOT Y gate, XOR gate** we can model some basic classifiers:

![](/images/IDL/MLP8.png)

Note. clearly the XOR needs to boundaries **(we call these decision boundaries)** \
Therefore, we say that the XOR cannot be modelled with just one perceptron

### Complex Decision Boundaries

If we create multiple decision boundaries, we can do the following:

- Find output of each decision boundary (i.e. does my point lie to the left or right of decision boundary)
- The above step happens in the hidden layer
- Then we can cumulate these decision boundary inputs
- From below fig. notice that only if sum == 5, the final neuron fires

![](/images/IDL/MLP9.png)

This way, we can model complex geometries, even complex ones like:

![](/images/IDL/MLP10.png)

### Another case for depth

Now, consider the above double pentagon figure. What if we were to do it
using just one layer?

We would have to approximate it using cylindrical regions (basically polygons with large number of sides, say 1000 sides)

![](/images/IDL/MLP11.png)

We can then use this cylinder decision boundary (multiples of them) to sort of make up
our double pentagon as shown below:

![](/images/IDL/MLP12.png)

But as seen above, **the major drawback is that the first layer will have an infinite
number of neurons!**

Now, comparing our depthwise vs spanwise solutions:
![](/images/IDL/MLP13.png)

## Sufficiency of Architecture
- A network arch is sufficient (i.e. sufficiently braod and sufficiently deep) it can
represent any function.

- Conversely if a network is not sufficient, it can miss out on information, and this
  lack of information can be propagated deeper causing major loss of information

  ![](/images/IDL/MLP15.png)
  In the above image, if the red lines our the first layer, the information passed to the
  second layer is that we are in those tiny diamond regions. However, we have no idea
  where we are in those diamonds. **(This is loss of info to the next layer!)**

  To mitigate this loss, instead of doing hard thresholding, we can use softer decision
  boundaries as shown below:

![](/images/IDL/MLP14.png)



# Further on MLPs

### Include bias as an input for simplifying downstream computations

|     Bias as a separate term            | Bias included in input               |
|:---------------------------------------|:-------------------------------------|
| ![](/images/IDL/MLP16.png)             | ![](/images/IDL/MLP17.png)           |

This also helps in simplifying the (z = Wx + b) equation from being affine to a 
linear form of (z = Wx)

## Proceeding from simple boolean functions

- We cannot handcraft our network like how we did for the double pentagon
- Therefore, we need a learnable method
- Also, most real functions are very complex and don't have nice visualizations
  like the [double pentagon](#complex-decision-boundaries)
- Therefore, we also need a way of learning such complex functions with only few samples
  and not relying on continuous data
- We do this by a sampling approach, where we calculate the error for every sample in
  our training data
  ![](/images/IDL/MLP18.png)


## The Perceptron algorithm

![](/images/IDL/MLP19.png)
![](/images/IDL/MLP20.png)

## Why is the perceptron algorithm not good?

### The primary issue is that the simple perceptron is flat and non-differentiable.

![](/images/IDL/MLP21.png)

### Data is never fully clean

We mostly never have nicely linearly separable data

![](/images/IDL/MLP22.png)

### The solution: Differentiable activations

![](/images/IDL/MLP23.png)

Now, making this activation differentiable has two benefits:
1. Let's us know if our changes is having a positive or negative effect on prediction
2. It allows us to do **backprop!**

![](/images/IDL/MLP24.png)

# Thinking about Derivatives

- Instead of thinking of derivatives as ```dy/dx``` where if we have y and x as vectors, dividing
  them would not make much sense, instead we define it as ```y' = alpha*x'```, where alpha is
  now a vector and alpha*x' can be though of as a dot product. Therefore, this alpha
  will now define the vector which when dot product with x gives the direction
  of the fastest increase in y.
  ![](/images/IDL/MLP27.png)

- Adavantage of doing it as ```y' = alpha*x'``` now is that for a multivariate form like above, 
  we can write the alpha vector as a partial derivate of y with x.
  ![](/images/IDL/MLP28.png)

- Now, we can clearly see how the gradient gives the direction of fastest increase in
  in the function. Therefore, if we want to minimize, we go in the direction exactly
  opposite to the gradient.
  ![](/images/IDL/MLP25.png)
  ![](/images/IDL/MLP26.png)


