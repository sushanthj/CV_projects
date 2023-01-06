---
layout: default
title: Intro to CNNs
parent: DL Overview
nav_order: 5
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

[Ref: 11-785](https://www.youtube.com/watch?v=VINm_uHUgF0&list=PLp-0K3kfddPzCnS4CqKphh-zT3aDwybDe&index=16&ab_channel=CarnegieMellonUniversityDeepLearning){: .btn .fs-3 .mb-4 .mb-md-0 }

# Simple method of achieving shift invariance

Assume we have a simple MLP which was trained to identify a flower. Now, if we run the MLP
blindly on the image, we will NOT have shift invariance.

A simple solution would be to scan the image at different positions and take the region which
gave the max output of an activation (say largest softmax class score).

![](/images/IDL4/scanMLP0.png)
![](/images/IDL4/scanMLP1.png)


## Important Backprop Theory
- Now, to train this, the initial 3 layers are seen to have the same weights (**shared weights**)
- Therefore treat it like a vector function where each and every window affects the final
  classification head. Now, if we want to backprop through such a function which depends
  on each and every window, we need to sum over the activations.

  ![](/images/IDL4/scanMLP2.png)
- Similaraly, the update step will also be such that the updated weights effect each and
  every one of the input weights equally as shown below:
  ![](/images/IDL4/scanMLP3.png)

## Summary

![](/images/IDL4/scanMLP5.png)

### Q. In which layer would you expect to see something that looks like a flower? \
Ans. Deeper Layers

![](/images/IDL4/scanMLP6.png)
![](/images/IDL4/scanMLP7.png)

### Q. Why do we need to distribute the scanning and not have one level of neurons scan the entire window at the same time?
Ans. It reduces the number of learnable parameters

![](/images/IDL4/mlpSCAN1.png)

However, to get a better understanding I strongly suggest going through the slides and video
links attached below to get a better understanding.

- In the references below, understand what the K,N,D,L terms represent
- If we consider a frequency spectrum of a voice recording, then considering some timestep
  we get an input vector of size 'LxD' where L = length of recording used and D = height here
  ![](/images/IDL4/timestep.png)

- Now, let's consider a case like this, where input vector of size L (here size 8) are 
  feeding into layer1 which has N1 neurons (4 neurons in below picture)
  ![](/images/IDL4/input_vector.png)
- We'll use this in a more generic scanning case as shwon below
  ![](/images/IDL4/scanning1.png)
  - From the above picture we calculate that each input vector has 8 timesteps (L)
  - Each vector has a dimensionality D (think height of the frequency plot image)
  - Therefore, number of weights connecting the input layer of size LD with 1st layer
    of size N1 leads to the first term LDN1
  - The next term is simply dependent on number of neurons in layer1 and layer2 = N1*N2

Now that we've seen the number of parameters in non-distributed scanning, let's compare
it to an example of non-distributed scanning as shown below:

![](/images/IDL4/scanning2.png)

Clearly, the distributed scanning has more shared computations which gives it fewer parameters. We also have many identical weights as shown below:

![](/images/IDL4/scanning3.png)
- In the above image, only the ones circled in greeen have unique parameters (weights)
  and the remaining are shared (thereby saving computation)
- **We also have a notion where saved computation has more gains over having more weights**

Now, if we think of the same logic in Image terms, it's just changing the dimensions
of the vectors (say K changes to a 2D patch which gets flattened to K^2)

#### Distributed vs Undistributed scanning for images

Note. Sometimes there's a (K + 1) term. That's just for the bias I'm assuming.

![](/images/IDL4/scanning4.png)
![](/images/IDL4/scanning5.png)

Finally, by doing distributed scanning we see the quantifiable effect as shown below:
![](/images/IDL4/scanning6.png)

#### Main Intuition

![](/images/IDL4/scanMLP7.png)

## Nice take on Max Pooling (Why is it needed?)

When we scan the image, if we find that one pixel which should belong to a petal has been
shifted, we somehow have to account for it. A nice solution is to not be too local focused
as to where the activation occured, but to just take the max of a small window.

By taking the max of the small window say 4x4, we're effectively not caring as to where the activation in that window occured, as long as it is within 4x4

### Nice consequence of above logic

- A little jitter in images can be expected due to irregularity of objects in the real world.
- However, in the speech world, jitter would mess up any phonetics that convey meaning
- As a result, in Speech recog there isn't much max-pooling


