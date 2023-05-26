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
the older states.

Here, marginalization is just a way of integrating probability density functions to get
a certainty in pose estimates of past poses. This works fine if the pose estimates were good
to begin with. This also cannot handle anything like loop closures.

[Detailed write-up](https://github.com/sushanthj/SLAM-and-Robot-Autonomy/blob/main/SLAM/EKF/SLAM_Assignment_2.pdf)
{: .btn .fs-5 .mb-4 .mb-md-0}

# Introduction

Here we'll use landmarks already known to us from the dataset (landmark poses) in 2D space.
Hence our localization would also be in 2D.

The below image shows the robot poses and the landmarks (in green).

![](/images/SLAM/EKF/Plot_with_grnd.png)

Since we localize in 2D our robot state space would also need to be in 2D. The robot's pose can
therefore be just *x, y, and yaw*.

Similar to the particle filter, here too we will have two steps in the algorithm:
- Prediction Step - **uses motion model**
- Update Step - **uses sensor model**

However, the update step in this case will depend on which landmarks we observe and will only
update the pose estimates of the robot and those landmarks alone.

The sensor model uses a laser rangefinder to give the landmark position in robot frame (which
is later converted into global coordinates)

# Robot State

Since we localize both **landmarks** and the **robot pose**, the state vector must contain
both information.

![](/images/SLAM/EKF/state_vector.png)

The above state vector captures robot pose and landmark position in **global coordinates**.
Robot pose has three variables (x,y,theta) and landmark position has two variables (x,y)
for each landmark.

This results in a **large state vector which is one of the drawbacks of EKF**.

## Motion Model

Previously, we used an odometry motion model in particle filters. This model is used here only
for comparison and explanation and *we will be using something slightly different.*

![](/images/SLAM/EKF/pf_motion_model.png)

Generally there are two forms of expressions for motion models:

|     Closed form calculation                                                         |                                                       Sampling                        |
|:------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
|   Calculates the robot's pose as a probability using the previous state as a prior  |   Calculates multiple poses where the robot might land based on the noise in model    |
| ![](/images/SLAM/EKF/closed_form.png)                                               | ![](/images/SLAM/EKF/sampling_motion.png)                                             |

In Particle Filters, we needed estimates of the where each particle would land and we therefore
had to use sampling. However, for EKF we need estimates of how noisy the estimated robot pose
would be. **Hence, we will kinda be using a closed form type.**


### 2D Robot Motion Model

- The robot will be constrained to only move in one axis (x-axis) shown as *d_t* in the robot's
body frame. However, it will be allowed to rotate along it's central axis by *alpha*.
- We will also add noise to motion in all three axis as *e_x, e_y, and e_alpha*

![](/images/SLAM/EKF/motion_model.png)

### Using Motion Model in Prediction

Here we mathematically define the prediction step used in EKF

#### Noise Free Prediction

![](/images/SLAM/EKF/noise_free_motion_pred.png)

#### Prediction with Uncertainty

We will now represent the prediction step as a non-linear function g(x,u) with some added noise

![](/images/SLAM/EKF/motion_pred_with_unc_1.png)

![](/images/SLAM/EKF/motion_pred_with_unc_2.png)

## Sensor Model

Given the range r and bearing β readings from a laser rangefinder, we can estimate the location of landmarks
in the global frame given a known robot state. We will use this as our measurement prediction model
h(p_t, β, r)

Therefore the landmark predictions are mathematically defined as:

![](/images/SLAM/EKF/landmark_pred_1.png)

The uncertainty is modelled separately as white noise (normally distributed with zero mean)

![](/images/SLAM/EKF/landmark_pred_2.png)

## Measurement Model

Using predicted state vector p t (contains robot pose and landmark) for the j’th landmark, the bearing **β**
and range **r** estimate for the j’th landmark is captured as **h(p_t , j)**

Where h(p_t , j) = measurement model

![](/images/SLAM/EKF/meas_model.png)



# EKF Algorithm

The main function of the EKF algorithm helps decrease the uncertainty in our state vector
(robot and landmark poses). Firstly, lets understand how the uncertainty in poses is captured.

Remember the [motion model](#motion-model) and [sensor model](#sensor-model).
We saw that noise was included in both cases. The motion model noise is called **control noise**
or process noise and the sensor model noise is called **measurement noise**

Since both control and measurement noise affects the robot state, it is easy to maintain a
combined covariance matrix P . The below equation shows how the state vector relates to the
mean and covariance matrices.

![](/images/SLAM/EKF/cov_intro.png)

In the EKF algorithm we will update parts the above covariance matrix in different steps:
- Given Control Reading : we update the Σ_xx primarily plus the first row and first column elements since they depend on x_t
- Given Sensor Reading : we update the whole covariance matrix (assuming we see all landmarks in each sensor reading)

## The Algorithm

![](/images/SLAM/EKF/algo.png)

## Finding the Covariances

- Step 4 of the [EKF Algorithm](#the-algorithm) is the **predicted covariance** based on
  some odometry reading.
- However, this predicted covariance has two parts. Robot pose covariance and Landmark covariance


### Robot Pose Covariance



