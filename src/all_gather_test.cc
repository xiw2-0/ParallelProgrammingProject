// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi
// test all gather


#include "all_gather.h"

int main(int argc, char *argv[])
{
  ::MPI_Init(&argc, &argv);
  int arr[10] = {0};
  int rank = 0;
  ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int val[] = {(rank + 2) * (9 + rank), (rank - 2)};
  
  auto ret_val = para::AllGather(val, 2, MPI_INT, arr, 2, MPI_INT, MPI_COMM_WORLD);
  printf("%d: ", rank);
  for (int i = 0; i < 10; ++i){
    printf("%d ", arr[i]);
  }
  printf("\n");
  return 0;
}
