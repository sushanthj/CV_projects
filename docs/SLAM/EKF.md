---
layout: default
title: EKF
parent: SLAM
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

# Extended Kalman Filter

We'll be using EKF to solve Non-linear online SLAM.

Note. Full SLAM = no marginalization (i.e. we optimize over all robot states). Online SLAM is
what we do in EKF where we only care about the previous robot state and marginalize out all
the old states.

Here, marginalization is just a way of integrating probability density functions to get
a certainty in pose estimates of past poses. This works fine if the pose estimates were good
to begin with. This also cannot handle anything like loop closures.

[Detailed write-up](https://github.com/sushanthj/SLAM-and-Robot-Autonomy/blob/main/SLAM/EKF/SLAM_Assignment_2.pdf)
{: .btn .fs-5 .mb-4 .mb-md-0}

# Introduction

Here we'll use landmarks already known to us from the dataset (landmark poses) in 2D space.
Hence our localization would also be in 2D

Since we localize in 2D our robot state space would also need to be in 2D.