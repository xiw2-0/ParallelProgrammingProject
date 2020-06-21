// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi

#include <cstdio>
#include <cassert>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#include "quick_sort.h"

void TestParallelQuickSort() {

  double arr[] = {1,9,0,8,89, 1092,7,-9,2,4,1,5,3,234,7,7,54};
  size_t size = sizeof(arr) / sizeof(double);

  std::vector<double> vec(arr, arr + size);

  para::ParallelQuickSort(arr, size);
  std::sort(vec.begin(), vec.end());
  for (int i = 0; i < vec.size(); ++i){
    assert(arr[i] == vec[i]);
  }
  printf("case #1 pass\n");

  int arr2[1000000] = {0};
  int arr22[1000000] = {0};

  std::mt19937 g(0);
  for (int i = 0; i < 1000000; ++i) {
    arr2[i] = g();
    arr22[i] = arr2[i];
  }

  auto time_start = std::chrono::system_clock::now();
  para::ParallelQuickSort(arr2, 1000000);
  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - time_start);
  printf("time for parallel sorting: %ld\n", dur.count());

  ::omp_set_num_threads(4);
  time_start = std::chrono::system_clock::now();
  std::sort(arr22, arr22+1000000);
  dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - time_start);
  printf("time for std sorting: %ld\n", dur.count());

  for (int i = 0; i < vec.size(); ++i){
    assert(arr2[i] == arr22[i]);
  }
  printf("case #2 pass\n");
}
