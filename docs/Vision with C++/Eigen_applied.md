---
layout: default
title: Eigen, OpenCV, and Images
parent: Computer Vision using C++
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

# Introduction

- Here we'll try to use Eigen alongside opencv to do some basic computer vision
- We'll emulate the eightpoint algorithm used to find the fundamental matrix in multiview
  geometry

# Basic Integration of Eigen in C++

I found that installing opencv is quite straightforward with the steps delineated below:

[OpenCV on Linux](https://www.geeksforgeeks.org/how-to-install-opencv-in-c-on-linux/){: .btn .fs-5 .mb-4 .mb-md-0 }

However, installing Eigen is a bit more tricky. I followed the below steps:

```bash
sudo apt update

sudo apt install libeigen3-dev

dpkg -S libeigen3-dev # only to verify if it's been installed right
```

However, if you use the following command, you can import eigen simply as: #include Eigen

```bash
sudo ln -sf eigen3/Eigen Eigen
```

# Example Code

## C++ File:

```cpp
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
// #include <inlucde/supp_two.hpp>

using namespace std;
using namespace Eigen;

int main()
{
    MatrixXf K1(3,3);
    MatrixXf K2(3,3);

    K1 << 1.5204e+03, 0.0000e+00, 3.0232e+02,
        0.0000e+00, 1.5259e+03, 2.4687e+02,
        0.0000e+00, 0.0000e+00, 1.0000e+00;

    K2 << 1.5204e+03, 0.0000e+00, 3.0232e+02,
        0.0000e+00, 1.5259e+03, 2.4687e+02,
        0.0000e+00, 0.0000e+00, 1.0000e+00;

    cout << "the expected fundamental matrix should be" << endl;

    cv::Mat im1;
    cv::Mat im2;
    im1 = cv::imread("/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_4/code/data/im1.png", 1);
    im2 = cv::imread("/home/sush/CMU/Assignment_Sem_1/CV_A/Assignment_4/code/data/im2.png", 1);

    /*
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", im1);
    cv::waitKey(0);
    */

    Eigen::MatrixXi pts1(10, 2);
    Eigen::MatrixXi pts2(10, 2);

    pts1 << 157, 231,
        309, 284,
        157, 225,
        149, 330,
        196, 316,
        302, 273,
        159, 324,
        158, 137,
        234, 340,
        240, 261;

    pts2 << 157, 211,
            311, 279,
            157, 203,
            149, 334,
            197, 318,
            305, 268,
            160, 327,
            157, 140,
            237, 346,
            240, 258;

    cout << pts1 << endl;
    cout << pts2 << endl;

    return 0;
}
```

## Header File - (place in include directory in the same folder as the .cpp file)

```cpp
#ifndef SUPP_TWO_HPP
#define SUPP_TWO_HPP

#include <eigen3/Eigen/Dense>

// Define and initialize the matrix
extern Eigen::MatrixXi pts1(110, 2);
extern Eigen::MatrixXi pts2(110, 2);

#endif
```

## CMakeLists.txt

```CMakeLists
cmake_minimum_required(VERSION 2.8)
project( eightpt )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})
add_executable( eightpt q2_1_in_cpp.cpp )
target_link_libraries( eightpt ${OpenCV_LIBS} )
```

## Execution

Run ```cmake .``` and ```make``` in the same level as the .cpp file