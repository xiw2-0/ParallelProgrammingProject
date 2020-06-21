// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi
// Parallel convolution, parallel pooling using OpenMP.

#ifndef MPI_CONVNET_OPS_H_
#define MPI_CONVNET_OPS_H_

#include <vector>

#include <mpi.h>



using std::vector;


namespace para {


// Convolution using MPI. Partition the features in tiles as the same sizes
// as filter, which is im2row. Then convolution is transformed into GEMV.
// Step size is set to 1.
//
// \param in_features input feature map
// \param filter convolution filter
// \param out_features the convolution result is stored in out_features
// \param comm MPI_Communicator used to calculate convolution
// \return true on success, false otherwise.
bool MPIConv(const vector<vector<double>>& in_features, const vector<vector<double>>& filter, int stride,
             vector<vector<double>>* out_features, ::MPI_Comm comm);


// Pooling using MPI. Partition the features in tiles as the same sizes
// as filter, which is im2row. Then max pooling is transformed into GEMV.
// The default step size is the size of filter.
//
// \param in_features input feature map
// \param filter_size the size of max pooling filter
// \param out_features the pooling result is stored in out_features
// \param comm MPI_Communicator used to calculate max pooling
// \return true on success, false otherwise.
bool MPIMaxPooling(const vector<vector<double>>& in_features, int filter_size,
                   vector<vector<double>>* out_features, ::MPI_Comm comm);


bool MPIGemv(const vector<vector<double>>& matrix, const vector<double>& in_vec,
             vector<double>* out_vec, ::MPI_Comm comm);

// Local gemv
void Gemv(const vector<vector<double>>& matrix, const vector<double>& in_vec,
          vector<double>* out_vec);

// im2row
void Im2row(const vector<vector<double>>& matrix, int filter_size, int stride,
            vector<vector<double>>* matrix_im2row);


bool MPIRowMax(const vector<vector<double>>& matrix,
               vector<double>* out_vec, ::MPI_Comm comm);
} // namespace para






#endif