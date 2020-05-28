// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi

#include <cstdio>
#include <cassert>

#include "miller_robin.h"

void TestMillerRobin() {
  assert(false == para::MillerRobin(-1, 4));
  ::printf("%d passed\n", -1);

  assert(false == para::MillerRobin(0, 4));
  ::printf("%d passed\n", 0);

  assert(false == para::MillerRobin(1, 4));
  ::printf("%d passed\n", 1);

  assert(true == para::MillerRobin(2, 4));
  ::printf("%d passed\n", 2);

  assert(false == para::MillerRobin(4, 4));
  ::printf("%d passed\n", 4);

  assert(true == para::MillerRobin(97, 4));
  ::printf("%d passed\n", 97);
  
  assert(false == para::MillerRobin(99, 4));
  ::printf("%d passed\n", 99);

  assert(true == para::MillerRobin(499, 4));
  ::printf("%d passed\n", 499);

  assert(false == para::MillerRobin(341, 4));
  ::printf("%d passed\n", 341);

  assert(false == para::MillerRobin(561, 4));
  ::printf("%d passed\n", 561);


  assert(false == para::MillerRobin(75361, 4));
  ::printf("%d passed\n", 75361);

  assert(true == para::MillerRobin(100000000063, 4));
  ::printf("%ld passed\n", 100000000063);
  
  assert(false == para::MillerRobin(46856248255981, 4));
  ::printf("%ld passed\n", 46856248255981);

}

