---
layout: default
title: Recap on Probability
parent: SLAM
nav_order: 1
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

[Ref Book. Probabalistic Robotics](https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf){: .btn .fs-3 .mb-4 .mb-md-0 }

# Discrete and Continuous Variables

## Discrete Variables and Notations

**Here the value X in P(X) can take on any value X=x_i, just that x_i are discrete points**

![](/images/SLAM/Probability_Review/1.png)

However, keep in my that we formally call the below function **Probability Mass Function**

![](/images/SLAM/Probability_Review/2.png)

## Continuous Random Variables

**Here the value of X in p(X) can take on a continuous variable X=x, where x is a smooth function**

Now, here we use lower_case ```p``` to denote the ```p(x)``` since in the continuous probability
world, we cannot speak in terms of absolute probability, but in terms of a density function:

![](/images/SLAM/Probability_Review/3.png)

## Get Absolute Probability from PDF

![](/images/SLAM/Probability_Review/4.png)

## Understanding p(x) PDF

As seen above, only the integration (area under curve) gives us the absolute probability. \
Therefore, this p(x) must be a curve of sorts, something like this:

![](/images/SLAM/Probability_Review/5.png)

## Can the value of p(x) > 1?

Yes. This is because p(x) is a PDF not absolute probability. \
Consider the example of a proximity sensor whose readings only range from 0m - 0.5m. The \
PDF for such a sensor would look like the below graph

![](/images/SLAM/Probability_Review/6.png)

# Joint and Contional Probability

## Joint Probability

Note. The calculation of absolute probability will change depending upon the nature of the 
variables:

![](/images/SLAM/Probability_Review/7.png)

## Conditional Probability

![](/images/SLAM/Probability_Review/8.png)


# Marginals and Conditionals

To start, lets get an intuition on what a marginal or conditional may look like:
- Let's consider a multivariate probability distribution (i.e there are say 2 random variables)
- Let us consider these two variables to have their own distributions
- Let these two distributions be *exam grades* and *study time*
- Imagine exam_grades are distributed along y-axis, and study_time along x-axis (sorry for asking you to imagine this much :/)
- Let the z-axis be a **joint probability** of both x and y
- Now, combining everything we should have a 3D plot

If we view this plot from the top view, we should see something like this:

![](/images/SLAM/Probability_Review/10.png)

## Crux of the Matter:

- Think of conditionals as taking a slice of this cloud and evaluating distribution of exam grades given a specific study time
- Think of marginals as squishing the cloud (say squishing all study-time data onto the exam-grades axis) and then studying the distribution

![](/images/SLAM/Probability_Review/11.png)

## Small Leap

Now that you've understood the intuition behind marginals, here's the math

![](/images/SLAM/Probability_Review/12.png)

# Bayes Theorem

![](/images/SLAM/Probability_Review/13.png)

