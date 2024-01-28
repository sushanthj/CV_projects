---
layout: default
title: Deep Learning Starter
parent: Deep Learning
nav_order: 2
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Before you Begin

My Primary Reference is my 11-785 Intro to Deep Learning at CMU

[Ref: CS231N](https://cs231n.github.io/linear-classify/){: .btn .fs-3 .mb-4 .mb-md-0 }
[Ref: CS229](https://www.youtube.com/watch?v=WViuTuAOPlM&list=PLoROMvodv4rNH7qL6-efu_q2_bPuy0adh&index=6&ab_channel=StanfordOnline){: .btn .fs-3 .mb-4 .mb-md-0 }
[Ref: CS231N Videos](https://www.youtube.com/watch?v=NfnWJUyUJYU&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC){: .btn .fs-3 .mb-4 .mb-md-0 }

# Era before DL

An [RI Seminar](https://www.youtube.com/watch?v=NIXA3ZTDI3Q) from 2013 of Deva Ramanan shows 
what was then the
State-of-the-Art methods in vision for object detection. I really like starting 
here as Deva had explained this transition from detection through classification of small 
sub-parts of a human (Deformable Parts Model) --->  to the advent of current Deep Learning

## Deformable Parts Model

### Naieve Way

- We have just one template which is found by taking the histogram-of-graidents (HOG)

- HOG is just a feature descriptor, just like SIFT and FAST are keypoint descriptors, but 
  HOG is on a more global level

- Use HOG template across image to find possible matches
![](/images/DL/deva_seminar_0.png)

### Issues with HOG

Now if we only used HOG templates for each class, it would have to capture the long tail
of all possible human poses as shown below:
![](/images/DL/deva_seminar2.png)
![](/images/DL/deva_seminar3.png)

### Deformable Templates
Therefore, to mitigate this issue of trying to capture all effects of the long tail, we can
instead only have templates for each part. 

- Define a template for a few parts separately
- Define certain learnable weights (think of it like springs whose lengths have to be learnt)
- Train the weights for these springs over a small dataset
- Develop other possible deformation modes using eigen vectors

![](/images/DL/deva_seminar_1.png)

### Conditional Independence

- Here, we don't have to think of it as K parts with L possible locations making it L^K configs
- Instead, we just construct as a dynamic programming model
- This will make use of the assumption that given a torso, the relationship of where the leg
  is w.r.t. the torso, i.e. the torso-leg relationship is independent of the torso-head 
  relationship. (This is called a spatial markov property)
- This conditional independece will help do inference in linear time

### Actual Training/Inference Steps:

- I.e we'll convolve each template across the image and get like a heat map for each of these
  templates
- Now that we have heat-maps, we'll relate them by the following formula (think of it like
  our objective function):
- ![](/images/DL/deva_seminar_deformation_scores.png)
  This formula says that we calculate local scores **(phi)** of each template
  (how well a head/face/hand) was matched in an image \
  as well as the contextual information of how far is a head location from an arm location 
  **(psi)** 

  The above formula can be thought of in the Dynamic Programming perspective where we have
  a graph with nodes of all possible head locations, all possible torso locations and we just
  need to find the least energy path
  ![](/images/DL/deva_seminar4.png)

  - This process shown in the computation graph is done actually on the images in the
    following manner:
    ![](/images/DL/deva_seminar5.png)

    Here, the steps are:
    1. We find the hotspots for the torso over the whole image (middle heat map above)
    2. Now, within a radius we want to find the possible location of the head (we can do
       this by taking the maxpool of a 2x2 location around the torso in a given radius)
    3. Therefore to do the above step we just take the heatmap of the head (first heat 
        map above) and do a maxpool. Then we shift this to match the ideal location of
        where the head should be w.r.t the torso
    4. By doing the above step of maxpool and shift, we have found one least energy path
       in the computational graph. **Therefore we have found for every torso in middle heat
       map, the score of every possible head it could connect to. Then we do the same process
       to compute the score of every possible torso and by chaining every possible best head
       location the legs can connect to**

- We can also run the test images over a few different models (these models would test
  the images for multiple deformation modes (like maybe 1/2 affine deformations, rotations etc))
  ![](/images/DL/deva_seminar6.png)

  
- Now, we can model the training of the above architechture as an SVM with hingle loss as:
  ![](/images/DL/deva_seminar_SVM.png)

- The main advantage in this method is that we won't have to create too huge a sample set of 
  negative samples (remember we have a huge number of possible locations where each feature 
  like hand or face can occur in the image)
    - Think of it, there will be a lot of images in the world which in a small window can look
    can look like a wrist. Therefore, huge negative sample set
    - However, in this case you will only care about a writst that is detected easily but
    it is also co-located near a face that was detected (we only care about the context here) \
    See table below: \

    |      Hard negatives without context    | Hard negatives in context            |
    |:---------------------------------------|:-------------------------------------|
    | ![](/images/DL/deva_seminar7.png)      | ![](/images/DL/deva_seminar8.png)    |


# Connecting Deformable Parts Model to Deep Learning
From what we saw above, to ideally detect a human we had to know both the local part-wise
detections and the global orientations. If you want to improve further, we could do even
sub-part detections and have more hierarchy. Therefore, one can see the need for hierarchial 
information to accurately detect objects. ---> this is a nice motivation for hierarchial
structure of deep networks as:

1. The first layer will see large features
2. As we go into deeper layers we will see more sub-part wise features
3. And finally we use all this information to guess where an object might be located in the image

![](/images/DL/deva_seminar9.png)
 
**TODO: This hierarchy of features may not be what's happening. Explain why!**

# Getting into DL

The first thing Deep Networks do is that they blur the line between extracting features and
actually doing classification on these features. (This happens throughout the network
and not only in the last layer of a network as people would commonly say)

![](/images/DL/DL1.png)

## Thinking of a feature extractor network as a big patch descriptor
Now, having said that, we could still use a network to extract some features and give an 
encoding how SIFT gives a 128 number encoding for an image patch (here the encoding
will be larger because the number of neurons in the final layer is low, and the outputs
of these neurons will also be low) maybe 500 number long encoding for the whole image

![](/images/DL/DL2.png)

## Simple Math on Fully Connected Networks

![](/images/DL/DL3.png)

![](/images/DL/Dl4.png)

# Convolutional Neural Nets

![](/images/DL/DL5.png)

We're going to claim that CNNs are just a special case of MLPs

## Difference between FCNs and CNNs

![](/images/DL/Dl6.png)

**Ans.** Because each neuron in the hidden layer is connected to every matrix in the input layer,
The size required would be *1M * 1M = 1e12* weights. Which is too much!

How do we fix this huge weight params issue?

1. Sparsity through local receptive fields: \
   Think of it as a feature detector (like edge detectors) which only looks at a 10x10 region \
   This will effectively make number of weights required as 1M * 10*10 = 100M
   ![](/images/DL/DL7.png)

2. Weight Sharing: \
   If we set the weights of all the above 10x10 receptive fields as the same, then we'll need
   **just 100 weights**

**Now, you can just call this an MLP (Multi-Layer Perceptron) with sparsity and weight sharing!**


### Q. Now, if the weights are shared for each receptive field, wouldn't that make each receptive field learn the same kind of features? How can we improve this?

Ans. 
- Instead of using the one 10x10 receptive field which has 100 weights and looks over the
whole image, we can instead have 10 different 10x10 receptive fields (each has 100 weights) 
which look over the image in the same way. i.e. we will have 10 convolutional filters

- And the total number of weights will still be low (100 * 10 = 1000 weights)

## Convolutional Layers

Now, let's define how exactly these convolutional filters work!

Firstly, remember that even in PyTorch we define the shape of an image as NCHW 

- N = number of images in one batch
- C = number of channels
- H,W = height and width of the image

![](/images/DL/DL8.png)
![](/images/DL/DL9.png)

*Note. Usually the filters to start with a small receptive field (that's why 3x3) and as the
network grows deeper, even if we continue using 3x3, because of the 'downsampling' nature of
the conv operations, we will end up increasing our receptive field.*

***This is also similar story to how the deformable parts model with smaller templates
 was better than having 1000s of large templates***

Therefore, it wouldn't make sense to start off with a really large filter! However it is now
common practice to have 5x5 filters in the first layer, and then 3x3 filters in all deeper 
layers. This is just emperical...

Now, if we add a bias term and as we discussed above add multiple filters (like a filter bank),
we get the following image:

![](/images/DL/DL10.png)

## Shift Invariance

[](/images/DL/DL11.png)

- If we zero pad an image in the first layer, the neuron in layer2 would see that the blue
triangle is closer to the image boundary. So you may think it starts associating that feature
to a particular location in the image.

- However, as we go deeper and deeper into the network, that same blue triangle will be
  more an more towards the center. The rest of the image will be all zeros

- In this case, the neurons see zeros in most places
- Therefore it can be said that padding actually does not allow the neurons to continuously
  learn any positional (aka spatial) dependence for the features. **This is what helps generate**
  **shift invariance!**
