// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi
// PageRank unit test

#include <cstdio>
#include <cassert>

#include "page_rank.h"

void TestPageRank() {
  para::PageRanker page_ranker(0.85, 1000, 0.00001);
  vector<vector<int>> connections {{0, 1}, {0, 2}, {0, 3},
                                   {1, 3}, {2, 4}, {3, 4},
                                   {1, 4}, {4, 0}};
  vector<double> page_rank_;
  vector<double> res_ {.296339, .113963, .113963, .162397, .313340};
  page_ranker.PageRank(connections, &page_rank_);
  

  assert(page_rank_.size() == res_.size());
  for (int i = 0; i < res_.size(); ++i) {
    assert(::abs(page_rank_[i] - res_[i]) < 0.00001);
  }
  printf("case #1 pass\n");


  vector<vector<int>> connections_1 {{0, 1}, {2, 1}};
  vector<double> res_1 {0.212766, .574468, .212766};
  page_ranker.PageRank(connections_1, &page_rank_);
  
  assert(page_rank_.size() == res_1.size());
  for (int i = 0; i < res_1.size(); ++i) {
    assert(::abs(page_rank_[i] - res_1[i]) < 0.00001);
  }
  printf("case #2 pass\n");
}