// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi
// \brief Test all unit test functions

#include <cstdio>

extern void TestMillerRobin();
extern void TestPageRank();

int main(int argc, char const *argv[]) {
  printf("=================Test starts=================\n\n");

  printf("Test MillerRobin...\n");
  TestMillerRobin();
  printf("\n");

  printf("Test PageRanker::PageRank...\n");
  TestPageRank();
  printf("\n");
  
  printf("=================Test ends=================\n");
  return 0;
}
