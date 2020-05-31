// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi
// Parallel quick sort using OpenMP.

#ifndef QUICK_SORT_H_
#define QUICK_SORT_H_

#include <vector>
#include <memory>
#include <omp.h>

using std::vector;


namespace para {

// when array size is smaller than MIN_SIZE, no more tasks will be created
const int MIN_SIZE = 1000;

template<typename T>
void Partition(T* arr, size_t size, size_t* partition_pos) {
  if (size <= 0) {
    return;
  }

  // maybe better to randomly choose a pivot
  T pivot_value = arr[0];

  size_t left = 0, right = size;
  while (left < right) {
    --right;
    while (left < right && arr[right] >= pivot_value) {
      --right;
    }
    if (left >= right) {
      break;
    }
    // swap
    arr[left] = arr[right];
    
    ++left;
    while (left < right && arr[left] <= pivot_value) {
      ++left;
    }
    if (left >= right) {
      break;
    }
    // swap
    arr[right] = arr[left];
  }
  arr[left] = pivot_value;

  *partition_pos = left;
}

template<typename T>
void ParallelSort(T* arr, size_t size) {
  if (size <= 1) {
    return;
  }

  size_t partition_pos = 0;
  Partition(arr, size, &partition_pos);

  if (size > MIN_SIZE) {
    #pragma omp taskgroup
    {
      if (partition_pos > 1){
        #pragma omp task mergable untied
        ParallelSort(arr, partition_pos);
      }
      if (partition_pos + 2 < size) {
        #pragma omp task mergable untied
        ParallelSort(arr + partition_pos + 1, size - partition_pos - 1);
      }
    }
  } else {
    #pragma omp task mergable untied
    {
      if (partition_pos > 1){
        ParallelSort(arr, partition_pos);
      }
      if (partition_pos + 2 < size) {
        ParallelSort(arr + partition_pos + 1, size - partition_pos - 1);
      }
    }
  }
}



// \brief Sort the data using parallel version of quicksort.
// Parallel is realized through OpenMP. For parallelism, we
// use omp task clause.
//
// For the 1st version, we only support data type that can use
// < to compare. And the sort is in place.
//
// \param arr the arr to be sorted
// \return void
template<typename T>
void ParallelQuickSort(T* arr, size_t size) {
  #pragma omp parallel num_threads(4)
  #pragma omp single
  {
    ParallelSort(arr, size);
  }
}



} // namespace para



#endif
