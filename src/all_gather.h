// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi
// Custom all gather using MPI send recv.

#ifndef ALL_GATHER_H_
#define ALL_GATHER_H_

#include <string>
#include <cstring>

#include <cstdlib>
#include <cstdio>

#include <vector>

#include <mpi.h>

using std::string;
using std::vector;

namespace para {

const int GATHER_TAG = 10101;
const int BCAST_TAG = 10102;

int Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
           int recvcount, MPI_Datatype recvtype, int root,
           MPI_Comm comm);


int Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);

// I try to use two steps to implement this method.
// This first step is gather to node 0, the second step is broadcast from node 0.
// Recently, only MPI_CHAR, MPI_INT, MPI_DOUBLE are supported.
int AllGather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
              void *recvbuf, int recvcount, MPI_Datatype recvtype,
              MPI_Comm comm);

} // namespace para



#endif