# ParallelProgrammingProject

This repo is used as my parallel programming project. There will be 3 programs
written using MPI, 3 written using OpenMP, and 1 using Hadoop MapReduce.

## OpenMP part

- Monte Carlo algorithm

I choose to implement Miller-Robin primality test. Miller-Robin test is a well-known
method to test whether a number is a prime or not. It belongs to Monte Carlo methods.

- PageRank

I use the power method to calculate the PageRank. The damping factor is set to 0.85 as
recommended. We have already know that most of pages have less than 10 out-links.
Therefore, it is very useful to represent the transfer probability matrix using spare
matrix. Here I choose the CSR format.

- Parallel quicksort

I use omp task clause to implement the parallelism of the quick sort. This work is
inspired by [three way quicksort](https://software.intel.com/content/www/us/en/develop/articles/an-efficient-parallel-three-way-quicksort-using-intel-c-compiler-and-openmp-45-library.html).

Similar to the page above, I set a threshold to avoid the tasks explosion. If there is no
such a threshold, the number of tasks will grow exponentially, which is not a good thing.
