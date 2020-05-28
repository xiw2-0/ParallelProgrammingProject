// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi
// \brief Test all unit test functions

#include <cstdio>

extern void TestMillerRobin();

int main(int argc, char const *argv[]) {
  printf("Test starts...\n");

  printf("Test MillerRobin()...\n");
  TestMillerRobin();
  printf("\n");
  
  printf("Test ends...\n");
  return 0;
}
