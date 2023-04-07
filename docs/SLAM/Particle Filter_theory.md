---
layout: default
title: Particle Filters Theory
parent: SLAM
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

Particle Filters are a direct application of Bayes Filters. The Short Version of Bayes Filter \
is shown below:

![](/images/SLAM/Particle_Filters/baye2.jpg)
![](/images/SLAM/Particle_Filters/baye3.jpg)
![](/images/SLAM/Particle_Filters/baye4.jpg)
![](/images/SLAM/Particle_Filters/baye5.jpg)
![](/images/SLAM/Particle_Filters/baye6.jpg)

# High Level Overview

We will be doing the following steps:

1. Initialize particles randomly across the whole map
2. resample at every step (selecting only the particles
  whose predicted laserscan matches actual laserscan) (low variance sampler)
3. Slowly decay the number of particles being randomly reinitialized
4. Use the mean of the particles's estimated pose to get true pose

[Detailed write-up](https://github.com/sushanthj/SLAM-and-Robot-Autonomy/blob/main/SLAM/Particle%20Filters/sushnatj_hw_1.pdf)
{: .btn .fs-5 .mb-4 .mb-md-0}