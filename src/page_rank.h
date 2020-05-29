// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi
// Calculate PageRank of a sparse big matrix.

#ifndef PAGE_RANK_H_
#define PAGE_RANK_H_

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>

#include <random>
#include <atomic>
#include <chrono>
#include <omp.h>


using std::vector;

namespace para {

class PageRanker {
 public:
  PageRanker(double damping_factor_, int max_iter_, double precision_);
  ~PageRanker() = default;

  // \brief Calculate PageRank value of a directed graph.
  //
  // OpenMP is used to speedup.
  //
  // \param connections a vector of {in_id, out_id} represents the connectivity of pages 
  // \param page_rank a vector of PageRank value for every node, it's the returning value
  // \return void
  void PageRank(const vector<vector<int>>& connections, vector<double>* page_rank);

 private:
  // \brief Convert a connection matrix into CSR format sparse matrix.
  //
  // \param connections a vector of {in_node, out_node} pair indicating the connectivity b/w nodes
  // \param csr_indptr represents rows of a sparse matrix using CSR format, returning param 
  // \param csr_indices represents col indices of non-zero values using CSR format, returning param
  // \param csr_data values of non-zeros, returning param
  //
  // CSR format: csr_indptr: csr_indptr[i+1] - csr_indptr[i] represens length of non-zeros in Row i
  //             csr_indices: non-zero's col position
  //             csr_data: contains non-zero's value 
  //
  void Connections2CSRMatrix(const vector<vector<int>>& connections, vector<int>* csr_indptr,
                             vector<int>* csr_indices, vector<double>* csr_data);

  // \brief Calculate PageRank vector using Power method.
  //
  // \param csr_indptr represents rows of a sparse matrix using CSR format 
  // \param csr_indices represents col indices of non-zero values using CSR format
  // \param csr_data values of non-zeros
  //
  // CSR format: csr_indptr: csr_indptr[i+1] - csr_indptr[i] represens length of non-zeros in Row i
  //             csr_indices: non-zero's col position
  //             csr_data: contains non-zero's value 
  //
  // \param page_rank a vector containing page rank values for every node
  // \return void
  void PowerMethodPR(const vector<int>& csr_indptr, const vector<int>& csr_indices,
                     const vector<double>& csr_data, vector<double>* page_rank);

 private:

  double damping_factor = 0.85;
  int max_iter = 29;
  double precision = 0.000000001;
};



} // namespace para



#endif