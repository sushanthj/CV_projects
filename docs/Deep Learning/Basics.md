---
layout: default
title: ML Basics
parent: Deep Learning
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

# Before you Begin

## Linear Algebra Review

![](/images/Intro_ML/uptill_logistic_regression/linear_alg_review.png)

![](/images/Intro_ML/uptill_logistic_regression/projection.png)

## Defining Linear Boundaries

The most basic equation of a line is ```y = mx + c```. This leads us to the formulation of a
baseline linear function to be:

```wx + b = 0``` which essentially defines a line

<img src="/images/Intro_ML/uptill_logistic_regression/linear_decision.png" width="50%"
height="50%">

# Perceptron

This basic algorithm is our intro to linear classifiers. The special part here is that it only
works on **sign(prediction)** and not on how good the actual prediction turns out.

![](/images/Intro_ML/uptill_logistic_regression/percep6.png)

## Algorithm

![](/images/Intro_ML/uptill_logistic_regression/percep1.png)

![](/images/Intro_ML/uptill_logistic_regression/percep2.png)

## Need for Intercept

![](/images/Intro_ML/uptill_logistic_regression/percep3.png)

## Summary

![](/images/Intro_ML/uptill_logistic_regression/percep4.png)