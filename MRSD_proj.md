---
layout: page
title: MRSD Capstone Project
permalink: /mrsd_proj/
nav_order: 11
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Introduction

I'm currently pursuing my Master's in Robotic Systems Development at Carnegie Mellon University.

![](/images/MRSD.png)

Our program is unique in that we emulate the systems engineering process which usually runs in
a robotics company. This process runs parallel to our capstone project where we build a robot,
secure funding and work within a budget, adhere to a timeline, and deliver on key performance
requirements.

I've been documenting my progress on the project work on a separate website which is accessed
by my peers occasionally.

[MRSD Project Website](https://mrsd-project.herokuapp.com/)
{: .btn .fs-5 .mb-4 .mb-md-0}


# SLAM in 2D

I implemented a version of AMCL particle filters tuned with more particles and larger resampling
radius to allow for accurate localization even in sparse maps.

![](/images/MRSD_proj/for_webpage-1.gif)