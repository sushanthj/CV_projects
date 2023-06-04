---
layout: default
title: Linear Algebra in Eigen
parent: Computer Vision using C++
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

# Background

Eigen is the numpy equivalent in C++. Here we look at some basic linear algebra computations
using eigen.

[Credits:](https://aleksandarhaber.com/starting-with-eigen-c-matrix-library/)

## Installation

To install eigen3 we can use the apt repository.

- ```sudo apt update```
- ```sudo apt install libeigen3-dev```
- Verify your installation by doing ```dpkg -L libeigen3-dev```

Then to use in code you simply need to include the following header file and work within the
following namespace:

```cpp
#include <eigen3/Eigen/Dense>

using namespace Eigen;
```

# Declaring and Defining Matrices

## Basics

Here we'll define a 3x3 matrix in two equivalent ways:

```cpp
#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
    // define 3x3 matrix -explicit declaration
    Matrix <float, 3, 3> matrixA;
    matrixA.setZero();
    cout << matrixA <<endl;

    // define 3x3 matrix -typedef declaration
    Matrix3f matrixA1;
    matrixA1.setZero();
    cout <<"\n"<<matrixA1<<endl;

    // Dynamic Allocation -explicit declaration
    Matrix <float, Dynamic, Dynamic> matrixB;

    // Dynamic Allocation -typedef declaration
    // 'X' denotes that the memory is to be dynamic
    MatrixXf matrixB1;

    // constructor method to declare matrix
    MatrixXd matrixC(10,10);

    // print any matrix in eigen is just piping to cout
    cout << endl << matrixC << endl;

    // resize any dynamic matrix
    MatrixXd matrixD1;
    matrixD1.resize(3, 3);
    matrixD1.setZero();
    cout << endl << matrixD1 << endl;

    return 0;
}
```

## Easier to Remember and Use

```cpp
int main()
{
    // directly init a matrix of zeros
    MatrixXf A;
    A = MatrixXf::Zero(3, 3);
    cout << "\n \n"<< A << endl;

    // directly init a matrix of ones
    MatrixXf B;
    B = MatrixXf::Ones(3, 3);
    cout << "\n \n"<< B << endl;

    // directly init a matrix filled with a constant value
    MatrixXf C;
    C = MatrixXf::Constant(3, 3, 1.2);
    cout << "\n \n"<< C << endl;

    // directly init identity (eye matrix)
    MatrixXd D;
    D = MatrixXd::Identity(3, 3);
    cout << "\n \n" << D << endl;

    MatrixXd E;
    E.setIdentity(3, 3);
    cout << "\n \n" << E << endl;
}
```

### Common Bug in above operations

```cpp
int main()
{
    MatrixXd V;
    V << 101, 102, 103, 104,
        105, 106, 107, 108,
        109, 110, 111, 112,
        113, 114, 115, 116;

    cout << V << endl;
}
```

- If you try to run the above code it will compile. However, in execution it will segfault.
- The reason will be that we did not allocate memory for the matrix V.

We can fix this by doing the following:

```cpp
int main()
{
    MatrixXd V;
    // option 1
    V.resize(4,4);

    // option 2
    V = MatrixXd::Zero(4, 4);

    V << 101, 102, 103, 104,
        105, 106, 107, 108,
        109, 110, 111, 112,
        113, 114, 115, 116;

    cout << V << endl;
}
```

## Explicitly Defining Matrix Entries

We already saw this above, but once we have defined the right shape of the matrix,
we can then define it's entries as shown below:

```cpp
MatrixXd V;
// option 1
V.resize(4,4);

V << 101, 102, 103, 104,
    105, 106, 107, 108,
    109, 110, 111, 112,
    113, 114, 115, 116;
```

## Slicing Matrices

```cpp
int main()
{
    MatrixXd V = MatrixXd::Zero(4,4);

    V << 101, 102, 103, 104,
        105, 106, 107, 108,
        109, 110, 111, 112,
        113, 114, 115, 116;

    cout << V << endl;

    MatrixXd Vblock = V.block(0, 0, 2, 2);
    cout << "\n \n" << Vblock << endl;
}
```

## Extracting Individual Rows and Columns

```cpp
int main()
{
    MatrixXd V = MatrixXd::Zero(4,4);

    V << 101, 102, 103, 104,
        105, 106, 107, 108,
        109, 110, 111, 112,
        113, 114, 115, 116;

    MatrixXd row1 = V.row(0);
    MatrixXd col1 = V.col(0);

    cout << row1 << endl;
    cout << col1 << endl;
}
```

The above is useful in finding the shape of any given matrix (like numpy.shape)

```cpp
#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

int main() {
    MatrixXd matrix(3, 4);  // Example matrix with 3 rows and 4 columns

    int numRows = matrix.rows();
    int numCols = matrix.cols();

    std::cout << "Number of rows: " << numRows << std::endl;
    std::cout << "Number of columns: " << numCols << std::endl;

    return 0;
}
```

