// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi
// Miller-Robin prime test.

#ifndef MILLER_ROBIN_H_
#define MILLER_ROBIN_H_

#include <cstdlib>
#include <cstdint>
#include <random>
#include <atomic>
#include <chrono>
#include <omp.h>

namespace para {

// \brief Test whether a number is a prime or not. Miller-Robin primality test.
// OpenMP is used to speedup.
// warning: assure that x,y,m < (2 ^ 62). Bigger values may overflow.
//
// \param candidate the number to be tested
// \param num_tests number of tests
// \return return true if the candidate is a prime, false if it is not a prime
bool MillerRobin(const int64_t candidate, const int num_tests);


// \brief Calculate exponential: x to the power of n, mod m.  
//
// warning: assure that x,y,m < (2 ^ 62). Bigger values may overflow.
//
// \param x base
// \param n exponential factor
// \param m modulo
// \return result of (x ^ n) % m
int64_t FastExponential(const int64_t x, const int64_t n, const int64_t m);


// \brief Calculate multiplication: x times y, mod m.  
//
// warning: assure that x,y,m < (2 ^ 62). Bigger values may overflow.
//
// \return result of (x * y) % m
int64_t FastMultiply(const int64_t x, const int64_t y, const int64_t m);

} // namespace para



#endif