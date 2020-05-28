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

#include <random>
#include <atomic>
#include <chrono>
#include <omp.h>

using std::vector;

namespace para {

// \brief Calculate PageRank value of a sparse matrix.
//
// OpenMP is used to speedup.
//
// \param matrix a sparse matrix which shows the connections b/w nodes
// \param page_rank a vector of PageRank value for every node, it's the returning value
// \return void
void PageRank(const vector<vector<int>>& matrix, vector<double>* page_rank);




} // namespace para



#endif