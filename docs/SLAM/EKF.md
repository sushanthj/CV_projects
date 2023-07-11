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

![](/images/SLAM/EKF/plot.png)

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

![](/images/SLAM/EKF/landmark_state_vector.jpeg)


In the EKF algorithm we will update parts of the above covariance matrix in different steps:
- The above image shows mean and covariance. <span style="color:black"> We find the mean using the non-linear functions of the motion and sensor model </span>
- <span style="color:black"> Given Control Reading </span> : we update the Σ_xx primarily plus the first row and first column elements since they depend on x_t
- <span style="color:black"> Given Sensor Reading  </span> : we update the whole covariance matrix (assuming we see all landmarks in each sensor reading)

## Algorithm Overview

![](/images/SLAM/EKF/algo.png)


## Setup and Initialization

Previously we saw the following image of what our state vector and covariance matrix would look

*Note:*
*state_vector = µ*

*covariance = Σ*


![](/images/SLAM/EKF/cov_intro.png)

![](/images/SLAM/EKF/landmark_state_vector.jpeg)

The same structre is followed in code where we will have **X** as our state vector and **P** as
our covariance matrix

```python
  # Generate variance from standard deviation
  sig_x2 = sig_x**2
  sig_y2 = sig_y**2
  sig_alpha2 = sig_alpha**2
  sig_beta2 = sig_beta**2
  sig_r2 = sig_r**2

  # Open data file and read the initial measurements
  data_file = open("../data/data.txt")
  line = data_file.readline()
  fields = re.split('[\t ]', line)[:-1]
  """
  The data file is extracted as arr and is given to init_landmarks below

  The data file has 2 types of entries:
  1. landmarks [β_1 r_1 β_2 r_2 · · · ], where β_i , r_i correspond to landmark
  2. control inputs in form of [d, alpha] (d = translation along x-axis)
  """
  arr = np.array([float(field) for field in fields])
  measure = np.expand_dims(arr, axis=1)
  t = 1

  # Setup control and measurement covariance
  control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
  measure_cov = np.diag([sig_beta2, sig_r2])

  # Setup the initial pose vector and pose uncertainty
  # pose vector is initialized to zero
  pose = np.zeros((3, 1))
  pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

  """
  measure = all landmarks
  measure_cov = known sensor covariance
  pose = initialized to (0,0)
  pose_cov = how much we trust motion model = fixed
  """
  k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                              pose_cov) # basically H_t in for-loop of pg 204
  print("Orig K is", k)

  # Setup state vector X by stacking pose and landmark states
  # X = [x_t, y_t, thetha_t, landmark1(range), landmark1(bearing), landmark2(range)...]
  X = np.vstack((pose, landmark))

  # Setup covariance matrix P by expanding pose and landmark covariances
  """
  - The covariance matrix for a state vector = [x,y,thetha] would be 3x3
  - However, since we also add landmarks into the state vector, we need to add that as well
  - Since there are 2*k landmarks, we create a new matrix encapsulating pose_cov and landmark_cov

  - this new cov matrix (constructed by np.block) is:

      [[pose_cov,        0     ],
        [    0,     landmark_cov]]

  """
  P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                [np.zeros((2 * k, 3)), landmark_cov]])

  print("Init covariance \n", np.round(P, 2))
```

### Landmark Initialization - Finding Landmark Pose Covariance

In the above code of setting up the state vector and covariance matrix, we see the following
lines:

```python
k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                              pose_cov) # basically H_t in for-loop of pg 204
                                                        # of probabalistic robotics

# covariance matrix
P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                [np.zeros((2 * k, 3)), landmark_cov]])
```

Here we see that the function init_landmarks is giving us the **landmark_cov and k**
(k = no. of landmarks). Let's take a look at this function to see it's working.

We will need to find the covariance of the landmarks by using the [Sensor Model](#sensor-model)

![](/images/SLAM/EKF/landmark_cov.png)


```python
def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    NOTE: Here we predict where the landmark may be, based on bearing and range sensor readings

    1. Number of landmarks (k)
    2. landmark states (just position (2k,1)) which will get stacked onto
       robot pose
    3. Covariance of landmark pose estimations (see theory)

    input1 init_measure    :  Initial measurements of form (beta0, l0, beta1,...) (2k,1)
    input2 init_measure_cov:  Initial covariance matrix (2, 2) per landmark given parameters.
    input3 init_pose       :  Initial pose vector of shape (3, 1).
    input4 init_pose_cov   :  Initial pose covariance of shape (3, 3) given parameters.

    return1 k              : Number of landmarks.
    return2 landmarks      : Numpy array of shape (2k, 1) for the state.
    return3 landmarks_cov  : Numpy array of shape (2k, 2k) for the uncertainty.
    '''

    k = init_measure.shape[0] // 2

    landmark = np.zeros((2*k, 1))
    landmark_cov = np.zeros((2*k, 2*k))

    x_t = init_pose[0][0]
    y_t = init_pose[1][0]
    theta_t = init_pose[2][0]

    # to find the covaraince of all landmark poses, we need to iterate over each landmark
    # and start filling in landmark_cov array initialized above. (we'll fill in diagonal
    # components only)
    for l_i in range(k):
        # l_i is the i'th landmark
        # init_measure.shape = (2k,1)
        beta = init_measure[l_i*2][0]
        l_range = init_measure[l_i*2 + 1][0]

        # need to find landmark location in global coords (l_x, l_y) to find H_l
        l_x = x_t + (float(l_range * np.cos(beta+theta_t)))
        l_y = y_t + (float(l_range * np.sin(beta+theta_t)))

        landmark[l_i*2][0] = l_x
        landmark[l_i*2+1][0] = l_y

        # Note, L here is the derivative of (l_x,l_y) vector (sensor model) w.r.t beta and theta
        # G_l is the derivative of same sensor model w.r.t state varialbes (x,y,theta)
        L = np.array([[float(-l_range* np.sin(beta+theta_t)), float(np.cos(beta+theta_t))],
                      [float(l_range* np.cos(beta+theta_t)), float(np.sin(beta+theta_t))]])

        # G_l represents the robot pose aspect of landmark measurement
        # therefore when measuring covariance, it will use robot pose covariance
        G_l = np.array([[1, 0, float(-l_range * np.sin(theta_t + beta))],
                        [0, 1, float(l_range * np.cos(theta_t + beta))]])

        # See theory, L below was derived w.r.t to measurement. Therefore,
        # during covariance calculation it will use measurement_covariance
        # Similarly, G defined w.r.t state variables (x,y,theta) therefore uses pose_covariance
        pred_landmark_cov = (G_l @ init_pose_cov @ G_l.T) + (L @ init_measure_cov @ L.T)

        assert(pred_landmark_cov.shape == (2,2))

        landmark_cov[l_i*2:l_i*2+2, l_i*2:l_i*2+2] = pred_landmark_cov

    return k, landmark, landmark_cov
```

## Main Loop of Algorithm

Once we have setup our state vector and covariance matrix, we start the series of
**prediction and update steps**. This is done in code as follows:

```python
# Core loop: sequentially process controls and measurements
for line in data_file:
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])

    # Control
    if arr.shape[0] == 2:
        print(f'{t}: Predict step')
        d, alpha = arr[0], arr[1]
        control = np.array([[d], [alpha]])

        X_pre, P_pre = predict(X, P, control, control_cov, k)

    # Measurement
    else:
        print(f'{t}: Update step')
        measure = np.expand_dims(arr, axis=1)

        X, P = update(X_pre, P_pre, measure, measure_cov, k)

        last_X = X
        t += 1
```

### Prediction Step

![](/images/SLAM/EKF/predict_graph.png)

After we initialize the state vector and covariance matrix, we can then proceed to the next
time step in our data file. At this stage we will be making predictions and updating the state
vector.

#### Finding the Covariance
- Step 4 of the [EKF Algorithm](#the-algorithm) is the **predicted covariance** based on
  some odometry reading.
- However, this predicted covariance has two parts. Robot pose covariance and Landmark covariance
- Landmark covariance was initialized in the previous step. Hence, <span style="color:blue">
  only the robot pose covariance and any covariance associated with robot pose will be updated.
  </span> This translates to only the first row and first column of the covariance matrix
  getting updated. See [cov matrix](#setup-and-initialization) for why first row and column

![](/images/SLAM/EKF/pose_cov.png)


### Prediction Step Code

Now, we will only be predicting the robot pose <span style="color:black"> (landmarks are initialized once and only updated
in sensor measurements i.e. update step)</span>

```python
def predict(X, P, control, control_cov, k):
    '''
    NOTE: Here we predict only the robot's new state (new state's mean and covariance)

    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance shape (3, 3) in the (x, y, theta) space.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''
    # TODO: Predict new position (mean) using control inputs (only geometrical, no cov here)
    theta_curr = X[2][0]

    d_t = control[0][0] # control input in robot's local frame's x-axis
    alpha_t = control[1][0]

    P_pred = deepcopy(P)
    pos_cov = deepcopy(P[0:3,0:3])

    X_pred = np.zeros(shape=X.shape)
    # update only robot pose (not landmark pose)
    X_pred[0][0] += float(d_t*np.cos(theta_curr))
    X_pred[1][0] += float(d_t*np.sin(theta_curr))
    X_pred[2][0] += float(alpha_t)

    X_pred = X_pred + X

    # TODO: Predict new uncertainity (covariance) using motion model noise, find G_t and R_t
    # NOTE: G_t needs to be mulitplied with P viz of shape (3 + 2k, 3+ 2k), because it has
    # pose and measurement cov. IN THIS STEP OF PREDICTION WE ONLY UPDATE POSE COV
    # Therefore G_t and R_t can be 3x3 (3 variables in state vector)

    G_t = np.array([[1, 0, float(-d_t * np.sin(theta_curr))],
                    [0, 1, float(d_t * np.cos(theta_curr))],
                    [0, 0,                1               ]])

    rotation_matrix_z = np.array([[float(np.cos(theta_curr)), -float(np.sin(theta_curr)), 0],
                                  [float(np.sin(theta_curr)),  float(np.cos(theta_curr)), 0],
                                  [           0,                            0,            1]])

    pose_pred_cov = (G_t @ pos_cov @ G_t.T) + \
                    (rotation_matrix_z @ control_cov @ rotation_matrix_z.T)

    # update just the new predicted covariance in robot pose, measurement pose is left untouched
    P_pred[0:3,0:3] = pose_pred_cov

    return X_pred, P_pred
```

### Update Step - A comparison step

![](/images/SLAM/EKF/update_graph.png)

The prediction step helps us get an esitmate of where the robot and landmarks should be.

Using this estimate of where the robot and landmarks are located, we check the difference
between what sensor reading we expect to get at these estimated poses
**(predicted sensor reading)** and where the robot actually is **(actual sensor reading)**

The method to predict a sensor reading **given an estimated robot and landmark pose** is by
defining a measurement model.


#### Measurement Model Definition

Using state vector p_t (contains robot pose and landmark) for the j’th landmark, the bearing **β**
and range **r** estimate for the j’th landmark is predicted as **h(p_t , j)**

Where h(p_t , j) = measurement model

![](/images/SLAM/EKF/meas_model.png)


#### Using the Measurement Model

The comparison step happens in the final parts of the algorithm shown below:

![](/images/SLAM/EKF/H_t.png)

![](/images/SLAM/EKF/update_1.png)

Here we see that we compare the predicted covariance Σ with H_t. Specifically the second
equation above is clear in it's purpose and is explained below:

- µ = µ̄ + K(z - ẑ)
- The RHS part of this equation shows how we compare the real sensor reading **z** and the
  predicted sensor reading **ẑ**
- The variable **K** then acts as a scaling factor only

#### Deriving H_t

![](/images/SLAM/EKF/jac_H.png)

### Update Step in Code

```python
def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    NOTE: Using predicted landmark & robot pose, we emulate our sensor reading using the sensor model

    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).

    Since we have a measurement, we will have to update both pose and measure covariances, i.e.
    the entire P_pre will be updated.

    Here we use the H_p and H_l described in the theory section. H_l and H_p will be combined
    to form H_t (the term in the EKF Algorithm in Probablistic Robotics). This H_t term
    will be defined for each landmark and stored in a massive matrix

    Q viz measurement covariance will need to be added to the H_t of each landmark, therefore
    it too will also be stored in a huge diagonal matrix
    '''
    # Q needs to be added to (Ht @ P_pre @ (Ht.T)) = (2*k, 2*k), therefore must be same shape
    Q = np.zeros(shape=(2*k, 2*k))

    # stack all predicted measurements into one large vector
    z_t = np.zeros(shape=(2*k, 1))

    # H_t as discussed above will be a large diagonal matrix where we'll stack H_p and H_l
    # side-by-side horizontally (making H_t 2x5 for each landmark). This will then be stacked
    # vertically, but again as a diagonal matrix.
    # H_t.T will also be multiplied with P_pre (3+2k, 3+2k). Therefore this needs to
    # also have 3+2k columns therefore the other dimensions should be 2k rows since
    # (H_p concat with H_l) = 2x5. Therefore, final H_t shape = 2k,3+2k
    H_t = np.zeros(shape=(2*k, 3+(2*k)))

    # iterate through every measurement, assuming every measurement captures every landmark
    num_measurements = k
    for i in range(num_measurements):
        # since we have a predicted pose already X_pre[0:3] we'll use that as our
        # linearization point

        # define the predicted pose of robot and landmark in global frame
        pos_x = X_pre[0][0] # robot pose_x in global frame
        pos_y = X_pre[1][0] # robot pose_y in global frame
        pos_theta = X_pre[2][0] #  bearing in global frame
        l_x = X_pre[3+i*2][0] # landmark i in global frame
        l_y = X_pre[4+i*2][0] # landmark i in global frame

        # convert predicted poses to local frame
        l_x_offset = l_x - pos_x
        l_y_offset = l_y - pos_y

        # use predicted pose of robot and landmark to get predicted measurements
        i_bearing = warp2pi(np.arctan2(l_y_offset, l_x_offset) - pos_theta) # bearing of i-th l
        i_range = math.sqrt(l_x_offset**2 + l_y_offset**2) # range of i-th landmark
        z_t[2*i][0] = i_bearing
        z_t[2*i+1][0] = i_range

        # Jacobian of measurement function (h(β,r) in theory) w.r.t pose (x,y,theta)
        # Note here we define h(β,r), whereas in theory it is h(r,β), hence rows are interchanged
        H_p = np.array([[(-l_x_offset/i_range)    , (-l_y_offset/i_range),       0],
                        [(l_y_offset/(i_range**2)), (-l_x_offset/(i_range**2)), -1],],
                        dtype=np.float64)

        # Note here we define h(β,r)
        H_l = np.array([[(l_x_offset/i_range)      , (l_y_offset/i_range)     ],
                        [(-l_y_offset/(i_range**2)), (l_x_offset/(i_range**2))]])

        # See theory how H_t is constructed. H_p goes only along the first three columns
        H_t[2*i : 2*i+2, 0:3] = H_p
        H_t[2*i : 2*i+2, 3+2*i : 5+2*i] = H_l

        Q[i*2:i*2+2, i*2:i*2+2] = measure_cov

    # Now after obtaining H_t and Q_t, find Kalman gain K
    # K = (3+2k, 3+2k) @ (3+2k, 2k) @ (2k, 2k) = (3+2k, 2k)
    K = P_pre @ H_t.T @ np.linalg.inv((H_t @ P_pre @ H_t.T) + Q)

    # Update pose(mean) and noise(covariance) using K
    X_updated = np.zeros(shape=X_pre.shape)
    X_updated = X_pre + (K @ (measure - z_t)) # (measure - z_t) = (actual - prediction)

    P_updated = (np.eye(2*k+3) - (K @ H_t)) @ P_pre

    return X_updated, P_updated
```

## Overview of Implementation

![](/images/SLAM/EKF/imp_1.png)

![](/images/SLAM/EKF/imp_2.png)

![](/images/SLAM/EKF/imp_3.png)

### Visual Interpretation

![](/images/SLAM/EKF/predict_graph.png)

![](/images/SLAM/EKF/update_graph.png)

![](/images/SLAM/EKF/update_3_graph.png)

![](/images/SLAM/EKF/update_4_graph.png)

## Detailed Notes and Derivation

[Detailed Derivation](https://drive.google.com/file/d/1sjxMO7rdThf_R7wtCv-aCYT6wKwvaGkB/view?usp=sharing)
{: .btn .fs-5 .mb-4 .mb-md-0}