---
layout: default
title: Least Squares SLAM
parent: SLAM
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

[Detailed write-up](https://github.com/sushanthj/SLAM-and-Robot-Autonomy/blob/main/SLAM/Non_linear_Least_Squares/sushantj_hw3.pdf)
{: .btn .fs-5 .mb-4 .mb-md-0}

# Background

In [EKF](/docs/SLAM/EKF) we used the jacobian of the measurement model in the update step.

![](/images/SLAM/EKF/H_t.png)

![](/images/SLAM/EKF/jac_H.png)

However, upon deeper inspection we see that this matrix is very sparse *(spare = lots of zeros)*

![](/images/SLAM/pose_graph/sparsity_explained.png)

It would therefore be most efficient if we could utilize this sparsity.

## Learning Goals

1. Learn to optimize operations by exploiting sparsity
2. Do Full SLAM using the Least Squares Formulation
   (EKF was only online SLAM where prior states were marginalized out)

# Introduction

Similar to the [EKF](/docs/SLAM/EKF) problem, we setup our state vector to comprise both
landmark and robot states. However, here we will make use the **Factor Graph formulation**
which consists of the following components:

- Factors (edges)
  - Odometry measurements
  - Landmark measurements

- States (nodes)
  - Robot poses
  - Landmark poses

![](/images/SLAM/pose_graph/factor_graph.png)

Here we’re already given the data of the all the factors and states present in the factor graph.
We will use this data to **minimize the variance in the predicted values of each measurement** (odometry measurements or
landmark measurements) between every two connected states on the factor graph.

This minimization will be crafted in a least squares minimization form. The high level procedure to do so
is shown below

![](/images/SLAM/pose_graph/least_squares_intuition.png)

Finally, the factor graph and least squares equivalence is seen below:

![](/images/SLAM/pose_graph/equivalence.png)

[Detailed Derivation Linear Least Sq.](https://drive.google.com/file/d/1YWcwMcVkOkE_xX38voyYF7nRC8N2zeMe/view?usp=sharing)
{: .btn .fs-5 .mb-4 .mb-md-0}

[Detailed Derivation Non-Linear Least Sq.](https://drive.google.com/file/d/1eJBy3T47wljiHbb_UeRwEOhBfJVlkbxQ/view?usp=sharing)
{: .btn .fs-5 .mb-4 .mb-md-0}

**We will be starting with 2D Linear SLAM and then moving onto 2D Non-Linear SLAM**

# 2D Linear SLAM

Here we will look to best fit our measurements **(z)** to our observed robot states **(θ)**.
Specifically, we will try to optimize over both the robot states and sensor measurements 
and try to reach the best middle ground. **Factor Graphs or Pose Graphs give us the 
best method of doing such global optimizations**.

This global optimization is first defined in the below manner in terms of increasing the
probability **p(z | θ)**

![](/images/SLAM/pose_graph/maximization.png)

#### Why Least Squares?

The reason we will formulate the above maximization equations as a least squares problem is
becuase we get mulitple landmark measurements as our robot moves, ending up with more
observations than variables (states in state vector). **Hence its not possible to extract an
exact solution which fits mathematically.** Therefore, our next easiest guess would be a least
squares formulation.

The exact derivation can be seen [Here](https://drive.google.com/file/d/1YWcwMcVkOkE_xX38voyYF7nRC8N2zeMe/view?usp=sharing). The abbreviated version is shown below.

## Derivation

Here we look to minimizing the uncertainty in our measurements (which are the edges of
our factor graph). To do so, we need to define measurement functions for these measurements.

## Odometry Measurement Function

![](/images/SLAM/pose_graph/odom_meas_func.png)

## Landmark Measurment Function

![](/images/SLAM/pose_graph/land_meas_func.png)

![](/images/SLAM/pose_graph/land_meas_func_2.png)

