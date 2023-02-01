---
layout: default
title: Expectation and Covariance
parent: SLAM
nav_order: 2
---

<details open markdown="block">
  <summary>
    Table of contents
  {: .text-delta }
  </summary>
1. TOC
{:toc}
</details>

Once we understand how probability density functions (PDFs) work, we can extend this to understand expectation

# Expectation

**It is a basically a weighted average (which is for discrete variables) for a continuous distribution**

1. If a random variable can have discrete outcomes, the probability of each outcome is weighted and an average is taken
2. In the continuous distribution sense, this becomes an integral each event of a random variable (x) and it's probability (p(x))

![](/images/SLAM/Probability_Review/14.png)

The above image shows some examples of how alpha(constant) and x(random varible) are computed \
However, there are a few basic properties to understand for all the Kalman Filters and Particle Filters we will study:

- E[alpha + x] = alpha + E[x]
- E[x,y] (called Joint Expectation)
- ![](/images/SLAM/Probability_Review/16.png) (called Conditional Expectation)
- E[x + y] = E[x] + E[y] (derived below)
- ![](/images/SLAM/Probability_Review/18.jpg)


## Correlation and Uncorrelation

### Uncorrelation and Joint Expectation
Above we mentioned Joint Expectation as E([x,y]. Now the only way we know x and y are uncorrelated here is if \
E[x,y] = = E[x]*E[y] (as x and y are clearly independent random variables)

**However, the inverse is not valid (i.e. if we only know they are uncorrelated, we cannot state independece like above equation)**

![](/images/SLAM/Probability_Review/17.png)

### Example to show that uncorrelation does not mean independence

![](/images/SLAM/Probability_Review/19.jpg)

# Connecting Variance to Expectation (I think it's important!)

![](/images/SLAM/Probability_Review/20.jpg)

# Covariances and Thinking Vectors

If we have a simple vector equation of the form: ![](/images/SLAM/Probability_Review/22.png)

Then for an equation of the form: ```y = Ax + b``` , we can find the covariance in terms of vector ```x``` as:

![](/images/SLAM/Probability_Review/23.png)

## Projecting Multivariate covariance

If ```z = f(x,y)```, then the covariance of z can be expressed as:

![](/images/SLAM/Probability_Review/21.jpg)

## Important Learning

![](/images/SLAM/Probability_Review/25.jpg)

## Properties of Covariance Matrix

![](/images/SLAM/Probability_Review/24.png)

### Properties of PSD

Note: In the second property below, he means if A is positive semi definite and B is positive definite
only then the sum will be positive definite. (At least one of them should be PSD and other PD)

![](/images/SLAM/Probability_Review/27.png)

# Correltation Coefficient

![](/images/SLAM/Probability_Review/26.jpg)