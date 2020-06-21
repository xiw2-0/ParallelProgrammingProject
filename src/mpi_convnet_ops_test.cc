// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi
// Test parallel convolution, parallel pooling and gemv.

#include "mpi_convnet_ops.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

void TestIm2row() {
  vector<vector<double>> mat1 {{1,2,3,1},{4,3,2,1},{3,1,2,8},{7,5,8,3}};
  vector<vector<double>> out1;
  vector<vector<double>> res1 {{1,2,4,3},{3,1,2,1},{3,1,7,5},{2,8,8,3}};
  para::Im2row(mat1, 2, 2, &out1);
  for (int i = 0; i < res1.size(); ++i) {
    assert(res1[i] == out1[i]);
  }
  ::printf("test case pass...\n");

}

void TestGemv() {
  vector<vector<double>> mat1 {{1,2,3,1},{4,3,2,1},{3,1,2,8},{7,5,8,3}};
  vector<double> vec1 {2,-9,3,8};
  vector<double> out1;
  vector<double> res1 {1,-5,67,17};
  para::Gemv(mat1, vec1, &out1);
  for (int i = 0; i < res1.size(); ++i) {
    assert(res1[i] == out1[i]);
  }
  ::printf("test case pass...\n");
}

void TestMPIGemv() {
  int rank = -1, n_process = 0;
  ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ::MPI_Comm_size(MPI_COMM_WORLD, &n_process);

  vector<vector<double>> matrix;
  vector<double> in_vec, out_vec;
  if (rank == 0) {
    vector<double> r {1, 9, 4};
    matrix.emplace_back(r);
    vector<double> r1 {2, 7, -1};
    matrix.emplace_back(r1);

    in_vec = std::move(vector<double> {1, 8, 3});
  }
  para::MPIGemv(matrix, in_vec, &out_vec, MPI_COMM_WORLD);
  if (rank == 0) {
    vector<double> ans {85, 55};
    assert(out_vec == ans);
    ::printf("test case #1 pass...\n");
  }


  if (rank == 0) {
    vector<vector<double>> mat1 {{1,2,3,1},{4,3,2,1},{3,1,2,8},{7,5,8,3}};
    vector<double> vec1 {2,-9,3,8};
    matrix.swap(mat1);
    in_vec.swap(vec1);
  }
  para::MPIGemv(matrix, in_vec, &out_vec, MPI_COMM_WORLD);
  if (rank == 0) {
    vector<double> ans {1,-5,67,17};
    assert(out_vec == ans);
    ::printf("test case #2 pass...\n");
  }
}

void TestMPIRowMax() {
  int rank = -1, n_process = 0;
  ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ::MPI_Comm_size(MPI_COMM_WORLD, &n_process);

  vector<vector<double>> matrix;
  vector<double> in_vec, out_vec;
  if (rank == 0) {
    vector<double> r {1, 9, 4};
    matrix.emplace_back(r);
    vector<double> r1 {2, 7, -1};
    matrix.emplace_back(r1);
  }
  para::MPIRowMax(matrix, &out_vec, MPI_COMM_WORLD);
  if (rank == 0) {
    vector<double> ans {9, 7};
    assert(out_vec == ans);
    ::printf("test case #1 pass...\n");
  }


  if (rank == 0) {
    vector<vector<double>> mat1 {{1,2,3,1},{4,3,2,1},{3,1,2,8},{7,5,8,3}};
    matrix.swap(mat1);
  }
  para::MPIRowMax(matrix, &out_vec, MPI_COMM_WORLD);
  if (rank == 0) {
    vector<double> ans {3, 4, 8, 8};
    assert(out_vec == ans);
    ::printf("test case #2 pass...\n");
  }
}

void TestMPIConv() {
  int rank = -1, n_process = 0;
  ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ::MPI_Comm_size(MPI_COMM_WORLD, &n_process);

  vector<vector<double>> matrix, filter, out_matrix;
  if (rank == 0) {
    matrix.emplace_back(vector<double> {1, 9, 4});
    matrix.emplace_back(vector<double> {2, 7, -1});

    vector<vector<double>> f {{1,-4},{2,7}};
    filter = std::move(f);
  }
  para::MPIConv(matrix, filter, 1, &out_matrix, MPI_COMM_WORLD);
  if (rank == 0) {
    vector<vector<double>> ans {{18, 0}};
    assert(out_matrix == ans);
    ::printf("test case #1 pass...\n");
  }
}

void TestMPIMaxPooling() {
  int rank = -1, n_process = 0;
  ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ::MPI_Comm_size(MPI_COMM_WORLD, &n_process);

  vector<vector<double>> matrix, out_matrix;
  if (rank == 0) {
    matrix.emplace_back(vector<double> {1, 9, 4, 0});
    matrix.emplace_back(vector<double> {2, 7, -1, 6});
  }
  para::MPIMaxPooling(matrix, 2, &out_matrix, MPI_COMM_WORLD);
  if (rank == 0) {
    vector<vector<double>> ans {{9, 6}};
    assert(out_matrix == ans);
    ::printf("test case #1 pass...\n");
  }


  if (rank == 0) {
    vector<vector<double>> mat1 {{1,2,3,1},{4,3,2,1},{3,1,2,8},{7,5,8,3}};
    matrix = std::move(mat1);
  }
  //out_matrix.clear();
  para::MPIMaxPooling(matrix, 2, &out_matrix, MPI_COMM_WORLD);
  if (rank == 0) {
    vector<vector<double>> ans {{4, 3},{7, 8}};
    assert(out_matrix == ans);
    ::printf("test case #2 pass...\n");
  }
}


int main(int argc, char *argv[]) {
  ::MPI_Init(&argc, &argv);
  int rank = -1;
  ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    printf("=================Test starts=================\n\n");

    printf("Test Im2row...\n");
    TestIm2row();
    printf("\n");

    printf("Test Gemv...\n");
    TestGemv();
    printf("\n");

    printf("Test MPIGemv...\n");
  }
  TestMPIGemv();

  if (!rank){
    printf("\n");
    printf("Test MPIRowMax...\n");
  }
  TestMPIRowMax();

  if (!rank){
    printf("\n");
    printf("Test MPIConv...\n");
  }
  TestMPIConv();

  if (!rank){
    printf("\n");
    printf("Test MPIMaxPooling...\n");
  }
  TestMPIMaxPooling();


  if (!rank)
    printf("=================Test ends=================\n");
  
  ::MPI_Finalize();
  return 0;
}


