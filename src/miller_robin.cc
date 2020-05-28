// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi

#include "miller_robin.h"

namespace para {

bool MillerRobin(const int64_t candidate, const int num_tests) {
  if (candidate == 2) {
    return true;
  }

  // even number or less than 2
  if (!(candidate & 1) || candidate < 2) {
    return false;
  }

  int64_t t = 0, u = candidate - 1;

  while(!(u & 1)) {
    t += 1;
    u >>= 1;
  }

  std::atomic_bool is_prime {true};

  #pragma omp parallel for
  for (int _ = 0; _ < num_tests; ++_) {
    // the candidate has been detected as not a prime
    if (!is_prime) {
      continue;
    }

    std::mt19937_64 generator(std::chrono::system_clock::now().time_since_epoch().count());
    int64_t rand_num = generator() % (candidate - 2) + 2;

    int64_t b = FastExponential(rand_num, u, candidate);

    if (b == 1) {
      continue;
    }

    bool pass = false;
    for (int i = 0; i <= t; ++i) {
      if (b == candidate - 1) {
        pass = true;
        break;
      }
      if (b < (1 << 31)) {
        b = (b * b) % candidate;
      } else {
        b = FastMultiply(b, b, candidate);
      }
    }
    if (!pass) {
      is_prime = false;
    }
  }
  
  return is_prime;
}

int64_t FastExponential(const int64_t x, const int64_t n, const int64_t m) {
  int64_t x_ = x % m;
  int64_t n_ = n % m;
  int64_t ans_ = 1;
  
  while (n_ > 0) {
    if (n_ & 1) {
      if (ans_ < (1 << 31) && x_ < (1 << 31)) {
        ans_ *= x_;
        ans_ %= m;
      } else {
        ans_ = FastMultiply(ans_, x_, m);
      } 
    }
    if (x_ < (1 << 31)) {
      x_ *= x_;
      x_ %= m;
    } else {
      x_ = FastMultiply(x_, x_, m);
    }
    n_ >>= 1;
  }

  return ans_;
}


int64_t FastMultiply(const int64_t x, const int64_t y, const int64_t m) {
  int64_t x_ = x % m;
  int64_t y_ = y % m;
  int64_t ans_ = 0;

  while (y_ > 0) {
    if (y_ & 1) {
      ans_ += x_;
      ans_ %= m;
    }
    x_ <<= 1;
    x_ %= m;
    y_ >>= 1;
  }
  return ans_;
}



} // namespace para


