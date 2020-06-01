// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi
// Custom all gather using MPI send recv.

#include "all_gather.h"


namespace para {


int Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
            int recvcount, MPI_Datatype recvtype, int root,
            MPI_Comm comm) {
  int rank = -1;
  ::MPI_Comm_rank(comm, &rank);
  
  int size;
  ::MPI_Comm_size(comm, &size);

  size_t type_size = 0;
  if (recvtype == MPI_INT) {
    type_size = sizeof(int);
  } else if (recvtype == MPI_DOUBLE) {
    type_size = sizeof(double);
  }else if (recvtype == MPI_CHAR) {
    type_size = sizeof(char);
  } else {
    printf("Data type not support recently\n");
    return -1;
  }

  int ret_val = 0;
  if (rank == root) {
    ::memcpy(recvbuf, sendbuf, recvcount * sizeof(recvtype));
    //
    // root recv msgs
    //
    for (int p = 1; p < size; ++p) {
      ::MPI_Status status;
      ret_val = ::MPI_Recv((char*)recvbuf + p * recvcount * type_size, recvcount,
                            recvtype, p, GATHER_TAG, MPI_COMM_WORLD, &status);
      if (ret_val != MPI_SUCCESS) {
        return ret_val;
      }
    }
  } else {
    //
    // others send msgs
    //
    ::MPI_Status status;
    ret_val = ::MPI_Send(sendbuf, sendcount, sendtype, 0, GATHER_TAG, MPI_COMM_WORLD);
  }
  return ret_val;
}

int Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  int rank = -1;
  ::MPI_Comm_rank(comm, &rank);
  
  int size;
  ::MPI_Comm_size(comm, &size);

  int ret_val = 0;
  if (rank == root) {
    //
    // root send msgs
    //
    for (int p = 1; p < size; ++p) {
      ::MPI_Status status;
      ret_val = ::MPI_Send(buffer, count, datatype, p, BCAST_TAG, MPI_COMM_WORLD);
      if (ret_val != MPI_SUCCESS) {
        return ret_val;
      }
    }
  } else {
    //
    // others recv msgs
    //
    ::MPI_Status status;
    ret_val = ::MPI_Recv(buffer, count,
                         datatype, 0, BCAST_TAG, MPI_COMM_WORLD, &status);
  }
  return ret_val;
}


int AllGather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               MPI_Comm comm) {
  int rank = -1;
  ::MPI_Comm_rank(comm, &rank);
  
  int size;
  ::MPI_Comm_size(comm, &size);

  int ret_val = 0;
  ret_val = Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, 0, MPI_COMM_WORLD);
  if (ret_val != MPI_SUCCESS) {
    return ret_val;
  }

  ret_val = Bcast(recvbuf, recvcount * size, recvtype, 0, MPI_COMM_WORLD);
  return ret_val;
}

} // namespace para

