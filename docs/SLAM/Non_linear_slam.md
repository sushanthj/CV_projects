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

- It would therefore be most efficient if we could utilize this sparsity.
- Additionally, EKF was sort of a local solver in that all prior states were
  marginalized out. Hence, we could not improve our estimates of our previous
  robot states even if we got more and more measurements.

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

# 2D Linear Least Squares SLAM

Here we will look to best fit our measurements **(z)** to our observed robot states **(θ)**.
Specifically, we will try to optimize over both the robot states and sensor measurements 
and try to reach the best middle ground. **Factor Graphs or Pose Graphs give us the 
best method of doing such global optimizations**.

This global optimization is first defined in the below manner in
terms of increasing probability  **p(z | θ)**

![](/images/SLAM/pose_graph/maximization.png)

#### Why Least Squares?

The reason we will formulate the above maximization equations as a least squares problem is
becuase we get mulitple landmark measurements as our robot moves, ending up with more
observations than variables (states in state vector). **Hence its not possible to extract an
exact solution which fits mathematically.** Therefore, our next easiest guess would be a least
squares formulation.

Think of it like simultaneous equations, if we have 3 variables, we need only 3 equations
to solve for the variables. However, if we have 4 equations, it's overdetermined. Similaraly,
here we have only few robot states, but a lot of observations constraining those states. These
observations can be odometry readings or sensor readings.

## Derivation

The exact derivation can be seen [in this link](https://drive.google.com/file/d/1YWcwMcVkOkE_xX38voyYF7nRC8N2zeMe/view?usp=sharing). The abbreviated version is shown below.

Now, let's assume that our measurement function **h(x)** is linear
<span style="color:pink"> (Even if it isn't linear, we approximate the
non-linear function by taking the first order taylor expansion later on) </span>

![](/images/SLAM/pose_graph/derivation_0.png)

![](/images/SLAM/pose_graph/derivation_1.png)

![](/images/SLAM/pose_graph/derivation_2.png)

Here we don't solve directly for **Ax = b** because of noise in the measurements. Instead,
we do the following:

![](/images/SLAM/pose_graph/derivation_filler_1.png)

The above equation will lead to: ![](/images/SLAM/pose_graph/derivation_3.png)

![](/images/SLAM/pose_graph/derivation_4.png)

![](/images/SLAM/pose_graph/derivation_5.png)

### Intuition for our final solution

![](/images/SLAM/pose_graph/solution_intuition.png)

## Measurement Funtion h(x)

<!-- Here we look to minimizing the uncertainty in our measurements (which are the edges of
our factor graph). To do so, we need to define measurement functions for these measurements. -->

### Odometry Measurement Function

![](/images/SLAM/pose_graph/odom_meas.png)

### Landmark Measurment Function

![](/images/SLAM/pose_graph/land_meas.png)


### Shapes of the A and b matrices

During the derivation we saw that

- ![](/images/SLAM/pose_graph/A_math.png)
- ![](/images/SLAM/pose_graph/b_math.png)

1. Now, the **A** matrix is seen to be the same size as the jacobian. We define the jacobian
   below (it's equated to A, but it's yet to be scaled by Σ^-0.5)
2. **b** is the same size as the number of measurements **z**. Hence, h_0 should also be a
   vector or the same size

![](/images/SLAM/pose_graph/A_b_shapes.png)

- We see that the rows of the jacobian are p, u1, u2, d1, d2 ..
- Since we're working in 2D, each of these measurements will have an *x* and *y* component
- p is technically not a measurement, but is the starting pose of the robot
- Hence we can define the number of rows of **A to be equal to (n_odom + 1) * 2 + n_obs * 2**

- We also see that the **b** vector has no. of rows = no. of measurements
- Therefore size of **b = n_poses * 2 + n_landmarks * 2**

## 2D Linear Least Squares in Code

### Fully Define the A and b matrices (setup step)

```python
def create_linear_system(odoms, observations, sigma_odom, sigma_observation,
                         n_poses, n_landmarks):
    '''
    \param odoms Odometry measurements between i and i+1 in the global coordinate system. Shape: (n_odom, 2).
    \param observations Landmark measurements between pose i and landmark j in the global coordinate system. Shape: (n_obs, 4).
    \param sigma_odom Shared covariance matrix of odometry measurements. Shape: (2, 2).
    \param sigma_observation Shared covariance matrix of landmark measurements. Shape: (2, 2).

    \return A (M, N) Jacobian matrix.
    \return b (M, ) Residual vector.
    where M = (n_odom + 1) * 2 + n_obs * 2, total rows of measurements.
          N = n_poses * 2 + n_landmarks * 2, length of the state vector.
    '''

    n_odom = len(odoms)
    n_obs = len(observations)

    M = (n_odom + 1) * 2 + n_obs * 2
    N = n_poses * 2 + n_landmarks * 2

    A = np.zeros((M, N))
    b = np.zeros((M, ))

    # Prepare Sigma^{-1/2}.
    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))

    # TODO: First fill in the prior to anchor the 1st pose at (0, 0)
    # The prior is just a reference frame, it also has some uncertainty, but no measurement
    # Hence the measurement function which estimates the prior is just a identity function
    # i.e h_p(r_t) = r_t. Since no measurements exist, the b matrix will have only zeros (already the case)

    # Here we also define the uncertainty in prior is same as odom uncertainty
    A[0:2, 0:2] = sqrt_inv_odom @ np.eye(2)

    # no need to update b (already zeros)

    # TODO: Then fill in odometry measurements
    """
    The A matrix structure is shown in the theory section. Along the rows, it has:
        - predicted prior (of size 1)
        - predicted odom measurements (of size n_odom)
        - predicted landmark measurements (of size n_obs)

    We will also follow the same order
    """

    H_odom = np.array([[-1,0,1,0], [0,-1,0,1]], dtype=np.float32)
    H_land = np.array([[-1,0,1,0], [0,-1,0,1]], dtype=np.float32)

    A_fill_odom = sqrt_inv_odom @ H_odom

    for i in range(n_odom):
        # declare an offset for i to include the prior term (which only occurs once along rows)
        j = i+1

        # A[2*j : 2*j+2 , 2*j : 2*j+4] = sqrt_inv_odom @ H_odom
        A[2*j : 2*j+2, 2*i : 2*i+4] = A_fill_odom
        b[2*j : 2*j + 2] = sqrt_inv_odom @ odoms[i]

    # TODO: Then fill in landmark measurements
    A_fill_land = sqrt_inv_obs @ H_land # H_land like H_odom is also a 2x4 matrix

    for i in range(n_obs):
        # observations = (52566,4) # (pose_index, landmark_index, measurement_x, measurement_y)
        # Therefore we need to check which pose is associated with which landmark
        p_idx = int(observations[i,0])
        l_idx = int(observations[i,1])
        # offset to account for prior (offset only along rows) + all odom measurements above
        j = i + n_odom + 1

        A[2*j : 2*j+2, 2*p_idx : 2*p_idx+2] = A_fill_land[0:2, 0:2]
        A[2*j : 2*j+2, 2*(n_poses + l_idx):2*(n_poses + l_idx)+2] = A_fill_land[0:2, 2:4]
        b[2*j : 2*j+2] = sqrt_inv_obs @ observations[i,2:4]

    # Convert matrix to sparse format which scipy can use
    return csr_matrix(A), b
```

### Solving the Linear System

The previous function has access to all the odometry and landmark measurements at once.
Therefore **we already have fleshed out the A and b matrices**. Using this, we can then
solve for the same as shown below:

```python
if __name__ == '__main__':

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odoms = data['odom']
    observations = data['observations']
    sigma_odom = data['sigma_odom']
    sigma_landmark = data['sigma_landmark']
    """
    The shapes of above values for 2d_linear.npz are:
    odoms = (999,2) which makes sense since there are 1000 robot poses
    observations = (52566,4) # (pose_index, landmark_index, measurement_x, measurement_y)
    sigma_odom = (2,2)
    sigma_landmark = (2,2)
    """

    # Build a linear system
    A, b = create_linear_system(odoms, observations, sigma_odom,
                                sigma_landmark, n_poses, n_landmarks)

    # Solve with the selected method
    for method in args.method:
        print(f'Applying {method}')

        total_time = 0
        total_iters = args.repeats
        for i in range(total_iters):
            start = time.time()
            x, R = solve(A, b, method)
            end = time.time()
            total_time += end - start
        print(f'{method} takes {total_time / total_iters}s on average')

        if R is not None:
            plt.spy(R)
            plt.show()

        traj, landmarks = devectorize_state(x, n_poses)

        # Visualize the final result
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)
```

As seen above, we have have multiple *methods* to solve the minimization function. Lookup
scipy.optimize.minimize to see a simple way in which one can solve an optimization problem
by defining an objective function. Here, our objective function is the **Ax = b**

The methods we will define below are pre-programmed to solve for such linear equations.

## Naieve Solvers

### Pseudo-Inverse

During our derivation we saw that one simple way to solve our minimization problem
is to use the psuedo-inverse:

![](/images/SLAM/pose_graph/pseudo_inv.png)

This is done in code as shown below:

```python
def solve_pinv(A, b):
    # TODO: return x s.t. Ax = b using pseudo inverse.
    N = A.shape[1]
    x = np.zeros((N, ))
    # Ax = b  <======> inv(A.T @ A) @ A.T @ A @ x = A.T b
    x = inv(A.T @ A) @ (A.T @ b)
    return x, None
```

#### Why not do SVD?

RECAP
- In CV ( see [Planar Homography](/Planar%20Homography) ) we saw that our final equation to
solve boiled down to the an Ax = 0 form.
- Here x is found by taking the SVD of A and choosing the eigen vector
  (with least eigen value) which forms the **null space of A**.
- Remember, null-space of a vector is the transformation (i.e. transformation matrix) which
  squeezed the vector onto a point (i.e. it reduces dimensions to zero).
- In this case **x** is the vector and we find the corresponding transformation matrix which
  forms it's null-space. This matrix then becomes our homography matrix
- For a better understanding of SVD, refer to [This Document](https://drive.google.com/file/d/1d6xcBbI2qcyaCx-rccI3sc9pdjQFMp2n/view?usp=sharing)


- In the SLAM problem, we **can't do SVD mainly because it would take too long to compute!**
- The vector here will have a million dimensions
- Also, we won't be utilizing the sparsity and letting go of an easy improvement

#### Scipy Default Solver

```python
def solve_default(A, b):
    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)
    return x, None
```

## Matrix Factorization - Utilizing the Sparsity

### Cholesky - LU Decomposition - Fast, little lower numerical stability

![](/images/SLAM/pose_graph/lu_1.png)
![](/images/SLAM/pose_graph/lu_2.png)
![](/images/SLAM/pose_graph/lu_3.png)
![](/images/SLAM/pose_graph/lu_4.png)
![](/images/SLAM/pose_graph/lu_5.png)
![](/images/SLAM/pose_graph/lu_6.png)

```python
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular

def solve_lu(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    # Better ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.SuperLU.html
    N = A.shape[1]
    x = np.zeros((N, ))

    lu = splu(A.T @ A, permc_spec='NATURAL')
    x = lu.solve(A.T @ b)

    U = eye(N)
    U = lu.U.A
    return x, U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutration_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    N = A.shape[1]
    x = np.zeros((N, ))

    lu = splu(A.T @ A, permc_spec='COLAMD')
    x = lu.solve(A.T @ b)

    U = eye(N)
    U = lu.U.A
    return x, U
```

Above we see an *lu_colamd* method defined separately:

#### Column Ordering

#### ChatGPT explanation
Sparse LU factorization is a technique used to solve systems of linear equations involving sparse matrices. The process involves decomposing the original matrix into a lower triangular matrix (L), an upper triangular matrix (U), and a permutation matrix (P) such that A = PLU. The resulting LU factorization can then be used to efficiently solve multiple linear systems with the same coefficient matrix.

The choice of column ordering can significantly affect the performance of the LU factorization algorithm. The COLAMD algorithm is an approximate minimum degree ordering method that aims to minimize the fill-in, which refers to the number of additional non-zero entries introduced during the LU factorization process. By reducing the fill-in, the COLAMD ordering can lead to faster factorization and improved computational efficiency.

#### My Understading
The colamd method rearranges (permutes) the A matrix to avoid accumulation of data in the last few
columns of one of the triangular matrices. This rearragement as seen in helps in the following manner:

- The algorithm reorders the columns of the matrix so that the non-zero entries are clustered together
- This can reduce the number of fill-in elements created during the factorization process.
- Therefore reordering can significantly reduce the computational and memory requirements of the LU
decomposition.

![](/images/SLAM/pose_graph/fill_in.png)

Some graphical examples which were computed for this dataset is shown below:

![](/images/SLAM/pose_graph/lu_sparse.png)

![](/images/SLAM/pose_graph/lu_colamd_sparse.png)

### QR Factorization - Slower, but more stable

![](/images/SLAM/pose_graph/qr_1.png)
![](/images/SLAM/pose_graph/qr_2.png)
![](/images/SLAM/pose_graph/qr_3.png)
![](/images/SLAM/pose_graph/qr_4.png)

```python
def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/theNded/PySPQR
    N = A.shape[1]
    x = np.zeros((N, ))
    R = eye(N)

    # rz gives the upper triangular part
    Z, R ,E, rank = rz(A, b, permc_spec='NATURAL')
    x = spsolve_triangular(R,Z,lower=False)

    return x, R


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/theNded/PySPQR
    N = A.shape[1]
    x = np.zeros((N, ))
    R = eye(N)

    # rz gives the upper triangular part
    Z, R ,E, rank = rz(A, b, permc_spec='COLAMD')

    # E is symmetric and is the permutation vector s.t. QR = AE
    E = permutation_vector_to_matrix(E)
    x = spsolve_triangular(R,Z,lower=False)
    x = E @ x

    return x, R
```

## Results of Least Sq. Optimization and Visual Inference

### Ground Truth Trajectory
![](/images/SLAM/pose_graph/ground_truth.png)

### Optimization Results

![](/images/SLAM/pose_graph/pinv_linear.png)

![](/images/SLAM/pose_graph/lu_colamd_linear.png)

![](/images/SLAM/pose_graph/linear_runtime_comp.png)

### Inference on the Runtime Comparisons

![](/images/SLAM/pose_graph/linear_inference_1.png)

![](/images/SLAM/pose_graph/linear_inference_2.png)

Therefore, it's safe to say that the **when A matrix is more dense, the colamd method makes
sense and efficiency boost is observed**. Therefore, for small A matrices, it might not be
necessary.

![](/images/SLAM/pose_graph/sparse_real_ex.png)

### Additionaly Optimization Results

![](/images/SLAM/pose_graph/linear_gt2.png)

![](/images/SLAM/pose_graph/linear_qr2.png)

# 2D Non-Linear Least Squares SLAM4

## Measurement Functions

![](/images/SLAM/pose_graph/non_linear_meas_func1.png)
![](/images/SLAM/pose_graph/non_linear_meas_func2.png)

## Building the Linear System from Non-Linear System

![](/images/SLAM/pose_graph/non_linear_system.png)

## Non Linear System in Code

```python
def warp2pi(angle_rad):
    """
    Warps an angle in [-pi, pi]. Used in the update step.
    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad = angle_rad - 2 * np.pi * np.floor(
        (angle_rad + np.pi) / (2 * np.pi))
    return angle_rad


def init_states(odoms, observations, n_poses, n_landmarks):
    '''
    Initialize the state vector given odometry and observations.
    '''
    traj = np.zeros((n_poses, 2))
    landmarks = np.zeros((n_landmarks, 2))
    landmarks_mask = np.zeros((n_landmarks), dtype=bool)

    for i in range(len(odoms)):
        traj[i + 1, :] = traj[i, :] + odoms[i, :]

    for i in range(len(observations)):
        pose_idx = int(observations[i, 0])
        landmark_idx = int(observations[i, 1])

        if not landmarks_mask[landmark_idx]:
            landmarks_mask[landmark_idx] = True

            pose = traj[pose_idx, :]
            theta, d = observations[i, 2:]

            landmarks[landmark_idx, 0] = pose[0] + d * np.cos(theta)
            landmarks[landmark_idx, 1] = pose[1] + d * np.sin(theta)

    return traj, landmarks


def odometry_estimation(x, i):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from (odometry between pose i and i+1)
    \return odom Odometry (\Delta x, \Delta y) in the shape (2, )
    '''
    # TODO: return odometry estimation
    odom = np.zeros((2, ))

    try:
        odom[0] = x[2*(i+1)] - x[2*i]
        odom[1] = x[2*(i+1)+1] - x[(2*i)+1]
    except:
        ipdb.set_trace()

    return odom


def bearing_range_estimation(x, i, j, n_poses):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return obs Observation from pose i to landmark j (theta, d) in the shape (2, )
    '''
    # TODO: return bearing range estimations
    obs = np.zeros((2, ))

    # given the robot pose and landmark location, get the bearing estimate (see theory)
    y_dist = x[(2*n_poses)+(2*j)+1] - x[(2*i)+1]
    x_dist = x[(2*n_poses)+(2*j)] - x[(2*i)]
    obs[0] = warp2pi(np.arctan2(y_dist, x_dist))
    obs[1] = np.sqrt(x_dist**2 + y_dist**2)

    return obs


def compute_meas_obs_jacobian(x, i, j, n_poses):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return jacobian Derived Jacobian matrix in the shape (2, 4)
    '''
    # TODO: return jacobian matrix
    jacobian = np.zeros((2, 4))

    y_dist = x[(2*n_poses)+(2*j)+1] - x[(2*i)+1]
    x_dist = x[(2*n_poses)+(2*j)] - x[(2*i)]

    sensor_range = np.sqrt(x_dist**2 + y_dist**2)

    jacobian[0,0] = y_dist/(sensor_range**2)
    jacobian[0,1] = -x_dist/(sensor_range**2)
    jacobian[0,2] = -y_dist/(sensor_range**2)
    jacobian[0,3] = x_dist/(sensor_range**2)

    jacobian[1,0] = -x_dist/sensor_range
    jacobian[1,1] = -y_dist/sensor_range
    jacobian[1,2] = x_dist/sensor_range
    jacobian[1,3] = y_dist/sensor_range

    return jacobian


def create_linear_system(x, odoms, observations, sigma_odom, sigma_observation,
                         n_poses, n_landmarks):
    '''
    \param x State vector x at which we linearize the system.
    \param odoms Odometry measurements between i and i+1 in the global coordinate system. Shape: (n_odom, 2).
    \param observations Landmark measurements between pose i and landmark j in the global coordinate system. Shape: (n_obs, 4).
    \param sigma_odom Shared covariance matrix of odometry measurements. Shape: (2, 2).
    \param sigma_observation Shared covariance matrix of landmark measurements. Shape: (2, 2).

    \return A (M, N) Jacobian matrix.
    \return b (M, ) Residual vector.
    where M = (n_odom + 1) * 2 + n_obs * 2, total rows of measurements.
          N = n_poses * 2 + n_landmarks * 2, length of the state vector.
    '''

    n_odom = len(odoms)
    n_obs = len(observations)

    M = (n_odom + 1) * 2 + n_obs * 2
    N = n_poses * 2 + n_landmarks * 2

    A = np.zeros((M, N))
    b = np.zeros((M, ))

    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))

    # TODO: First fill in the prior to anchor the 1st pose at (0, 0)
    # The prior is just a reference frame, it also has some uncertainty, but no measurement
    # Hence the measurement function which estimates the prior is just a identity function
    # i.e h_p(r_t) = r_t. Since no measurements exist, the b matrix will have only zeros (already the case)

    # Here we also define the uncertainty in prior is same as odom uncertainty
    A[0:2, 0:2] = sqrt_inv_odom @ np.eye(2)

    # no need to update b (already zeros)

    H_odom = np.array([[-1,0,1,0], [0,-1,0,1]], dtype=np.float32)
    A_fill_odom = sqrt_inv_odom @ H_odom

    # TODO: Then fill in odometry measurements
    for i in range(n_odom):
        # declare an offset for i to include the prior term (which only occurs once along rows)
        j = i+1

        A[2*j : 2*j+2, 2*i : 2*i+4] = A_fill_odom
        b[2*j : 2*j + 2] = sqrt_inv_odom @ (odom[i] - odometry_estimation(x,i))

    # TODO: Then fill in landmark measurements
    for i in range(n_obs):
        p_idx = int(observations[i,0])
        l_idx = int(observations[i,1])
        Al = sqrt_inv_obs @ compute_meas_obs_jacobian(x, p_idx, l_idx, n_poses)

        # offset again to account for prior
        j = n_odom+1+i
        A[2*j:2*j+2, 2*p_idx:2*p_idx+2] = Al[0:2,0:2]
        A[2*j:2*j+2,2*(n_poses+l_idx):2*(n_poses+l_idx)+2] = Al[0:2,2:4]

        b[2*j:2*j+2] = sqrt_inv_obs @ warp2pi(observations[i,2:4] - bearing_range_estimation(x,p_idx,l_idx,n_poses))

    return csr_matrix(A), b


if __name__ == '__main__':

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odom = data['odom']
    observations = data['observations']
    sigma_odom = data['sigma_odom']
    sigma_landmark = data['sigma_landmark']

    # Initialize: non-linear optimization requires a good init.
    for method in args.method:
        print(f'Applying {method}')
        traj, landmarks = init_states(odom, observations, n_poses, n_landmarks)
        print('Before optimization')
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)

        # Iterative optimization
        x = vectorize_state(traj, landmarks)
        for i in range(10):
            A, b = create_linear_system(x, odom, observations, sigma_odom,
                                        sigma_landmark, n_poses, n_landmarks)
            dx, _ = solve(A, b, method)
            x = x + dx
        traj, landmarks = devectorize_state(x, n_poses)
        print('After optimization')
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)

```

## Results

![](/images/SLAM/pose_graph/non_linear_results_1.png)
![](/images/SLAM/pose_graph/non_linear_results_2.png)

## Distinction between Non-Linear and Linear Optimization

- The Non Linear optimization process is different in that we have a residual h 0 term which causes the
  optimizer to solve for the error between predicted and actual measurement. In the linear case the optimizer
  directly solves for x in the equation Ax = b.
- Additionally, the non-linear method also requires a good initial estimate to start. This is because the
  optimization is iterative for the non-linear case if the initialization is bad it will take longer or not converge
  within the required error tolerance.