// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi
// Parallel convolution, parallel pooling using OpenMP.

#include "mpi_convnet_ops.h"

#include <cstdio>
#include <mpi.h>

#define BLOCK_LOW(id, np, m) ((id)*(m)/(np))
#define BLOCK_HIGH(id, np, m) ((id) == (np)-1 ? (m)-1 : ((BLOCK_LOW((id)+1, np, m))-1))
#define BLOCK_SIZE(id, np, m) (BLOCK_HIGH(id, np, m) - BLOCK_LOW(id, np, m) + 1)


namespace para {




bool MPIConv(const vector<vector<double>>& in_features, const vector<vector<double>>& filter, int stride,
             vector<vector<double>>* out_features, ::MPI_Comm comm) {
  out_features->clear();

  int rank = -1;
  ::MPI_Comm_rank(comm, &rank);

  vector<vector<double>> matrix_im2row;
  vector<double> filter_im2row;
  int in_height, in_width, filter_size;

  if (!rank) {
    // filter should be a square
    if (filter.size() == 0 || filter[0].size() != filter.size()) {
      return false;
    }

    // make sure sizes of feature maps are appropriate
    if (in_features.size() == 0 || in_features[0].size() == 0) {
      return false;
    }

    in_height = in_features.size();
    in_width = in_features[0].size();
    filter_size = filter.size();

    //
    // apply im2row to feature map
    //
    Im2row(in_features, filter_size, stride, &matrix_im2row);

    //
    // apply im2row to filter
    //
    for (int f_row = 0; f_row < filter_size; ++f_row) {
      for (int f_col = 0; f_col < filter_size; ++f_col) {
        filter_im2row.emplace_back(filter[f_row][f_col]);
      }
    }
    

  }
  

  //
  // gemv
  //
  vector<double> out_vec;
  MPIGemv(matrix_im2row, filter_im2row, &out_vec, comm);

  if(!rank) {
    // reshape output vector
    int out_features_width = (in_width - filter_size) / stride + 1;
    for (int i_row = 0; i_row < (in_height - filter_size) / stride + 1; ++i_row) {
      int row_start = i_row * out_features_width;

      vector<double> out_features_row(out_vec.begin() + row_start, out_vec.begin() + row_start + out_features_width);
      out_features->emplace_back(std::move(out_features_row));
    }
  }
  return true;
}

bool MPIMaxPooling(const vector<vector<double>>& in_features, int filter_size,
                   vector<vector<double>>* out_features, ::MPI_Comm comm) {
  out_features->clear();

  int rank = -1;
  ::MPI_Comm_rank(comm, &rank);

  int in_height, in_width, stride;
  vector<vector<double>> matrix_im2row;

  if (!rank) {
    // make sure sizes of feature maps are appropriate
    if (in_features.size() == 0 || in_features[0].size() == 0) {
      return false;
    }

    in_height = in_features.size();
    in_width = in_features[0].size();
    stride = filter_size;

    //
    // apply im2row to feature map
    //
    Im2row(in_features, filter_size, stride, &matrix_im2row);
  }


  //
  // max pooling
  //
  vector<double> out_vec;
  MPIRowMax(matrix_im2row, &out_vec, comm);

  if (!rank) {
    // reshape output vector
    int out_features_width = (in_width - filter_size) / stride + 1;
    for (int i_row = 0; i_row < (in_height - filter_size) / stride + 1; ++i_row) {
      int row_start = i_row * out_features_width;

      vector<double> out_features_row(out_vec.begin() + row_start, out_vec.begin() + row_start + out_features_width);
      out_features->emplace_back(std::move(out_features_row));
    }
  }


  return true;
}


bool MPIGemv(const vector<vector<double>>& matrix, const vector<double>& in_vec,
             vector<double>* out_vec, ::MPI_Comm comm) {
  out_vec->clear();

  int rank = -1, n_process = 0;
  ::MPI_Comm_rank(comm, &rank);
  ::MPI_Comm_size(comm, &n_process);

  // using rowwise block decomposition

  //
  // master send matrix size to all nodes
  //
  // matrix_size[0]: height, [1]: width
  int matrix_size[2] = {-1, -1};
  if (rank == 0) {
    matrix_size[0] = matrix.size();
    matrix_size[1] = matrix[0].size();
  }
  ::MPI_Bcast(&matrix_size, 2, MPI_INT, 0, comm);

  //
  // master send matrix data and vector data to all nodes
  //
  int n_rows = BLOCK_SIZE(rank, n_process, matrix_size[0]);
  
  if (rank == 0) {  // master
    // broadcast vector
    vector<double> vec(in_vec);
    ::MPI_Bcast(vec.data(), matrix_size[1], MPI_DOUBLE, 0, comm);
    MPI_Status status;

    // distribute matrix
    for (int i_process = 1; i_process < n_process; ++i_process) {
      n_rows = BLOCK_SIZE(i_process, n_process, matrix_size[0]);
      int row_start = BLOCK_LOW(i_process, n_process, matrix_size[0]);
      for (int i_row = 0; i_row < n_rows; ++i_row) {
        ::MPI_Send(matrix[row_start + i_row].data(), matrix_size[1], MPI_DOUBLE, i_process, 0, comm);
      }
    }

    // calculate
    out_vec->assign(matrix_size[0], 0.0);
    n_rows = BLOCK_SIZE(0, n_process, matrix_size[0]);
    for (int i_row = 0; i_row < n_rows; ++i_row) {
      double res = 0.0;
      for (int i_vec = 0; i_vec < in_vec.size(); ++i_vec) {
        res += matrix[i_row][i_vec] * in_vec[i_vec];
      }
      (*out_vec)[i_row] = res;
    }

    // merge results
    for (int i_process = 1; i_process < n_process; ++i_process) {
      n_rows = BLOCK_SIZE(i_process, n_process, matrix_size[0]);
      int row_start = BLOCK_LOW(i_process, n_process, matrix_size[0]);
      ::MPI_Recv(out_vec->data() + row_start, n_rows, MPI_DOUBLE, i_process, 1, comm, &status);
    }
    return true;
  } else { // workers

    // recv vector from master
    vector<double> vec(matrix_size[1], 0.0);
    ::MPI_Bcast(vec.data(), matrix_size[1], MPI_DOUBLE, 0, comm);
    
    // recv part of matrix
    vector<vector<double>> sub_mat;
    MPI_Status status;
    for (int i_row = 0; i_row < n_rows; ++i_row) {
      vector<double> mat_row(matrix_size[1], 0.0);
      ::MPI_Recv(mat_row.data(), matrix_size[1], MPI_DOUBLE, 0, 0, comm, &status);
      sub_mat.emplace_back(std::move(mat_row));
    }

    // calculate
    vector<double> out_sub_vec;
    Gemv(sub_mat, vec, &out_sub_vec);

    // send back the results to master
    ::MPI_Send(out_sub_vec.data(), out_sub_vec.size(), MPI_DOUBLE, 0, 1, comm);

    return true;
  }
}

bool MPIRowMax(const vector<vector<double>>& matrix,
               vector<double>* out_vec, ::MPI_Comm comm) {
  out_vec->clear();

  int rank = -1, n_process = 0;
  ::MPI_Comm_rank(comm, &rank);
  ::MPI_Comm_size(comm, &n_process);

  //
  // master send matrix size to all nodes
  //
  // matrix_size[0]: height, [1]: width
  int matrix_size[2] = {-1, -1};
  if (rank == 0) {
    matrix_size[0] = matrix.size();
    matrix_size[1] = matrix[0].size();
  }
  ::MPI_Bcast(&matrix_size, 2, MPI_INT, 0, comm);

  //
  // master send matrix data to all nodes
  //
  int n_rows = BLOCK_SIZE(rank, n_process, matrix_size[0]);
  
  if (rank == 0) {  // master
    MPI_Status status;

    // distribute matrix
    for (int i_process = 1; i_process < n_process; ++i_process) {
      n_rows = BLOCK_SIZE(i_process, n_process, matrix_size[0]);
      int row_start = BLOCK_LOW(i_process, n_process, matrix_size[0]);
      for (int i_row = 0; i_row < n_rows; ++i_row) {
        ::MPI_Send(matrix[row_start + i_row].data(), matrix_size[1], MPI_DOUBLE, i_process, 0, comm);
      }
    }

    // calculate
    out_vec->assign(matrix_size[0], 0.0);
    n_rows = BLOCK_SIZE(0, n_process, matrix_size[0]);
    for (int i_row = 0; i_row < n_rows; ++i_row) {
      double res = matrix[i_row][0];
      for (auto row_element : matrix[i_row]) {
        if (res < row_element) {
          res = row_element;
        }
      }
      (*out_vec)[i_row] = res;
    }

    // merge results
    for (int i_process = 1; i_process < n_process; ++i_process) {
      n_rows = BLOCK_SIZE(i_process, n_process, matrix_size[0]);
      int row_start = BLOCK_LOW(i_process, n_process, matrix_size[0]);
      ::MPI_Recv(out_vec->data() + row_start, n_rows, MPI_DOUBLE, i_process, 1, comm, &status);
    }
    return true;
  } else { // workers
    
    // recv part of matrix
    vector<vector<double>> sub_mat;
    MPI_Status status;
    for (int i_row = 0; i_row < n_rows; ++i_row) {
      vector<double> mat_row(matrix_size[1], 0.0);
      ::MPI_Recv(mat_row.data(), matrix_size[1], MPI_DOUBLE, 0, 0, comm, &status);
      sub_mat.emplace_back(std::move(mat_row));
    }

    // calculate
    vector<double> out_sub_vec;
    for (const auto& row : sub_mat) {
      double res = row[0];
      for (auto row_element : row) {
        if (res < row_element) {
          res = row_element;
        }
      }
      out_sub_vec.emplace_back(res);
    }

    // send back the results to master
    ::MPI_Send(out_sub_vec.data(), out_sub_vec.size(), MPI_DOUBLE, 0, 1, comm);

    return true;
  }

}

void Gemv(const vector<vector<double>>& matrix, const vector<double>& in_vec,
          vector<double>* out_vec) {
  out_vec->clear();
  
  for (const auto& row : matrix) {
    double res = 0.0;
    for (int i_vec = 0; i_vec < in_vec.size(); ++i_vec) {
      res += row[i_vec] * in_vec[i_vec];
    }
    out_vec->emplace_back(res);
  }
}

void Im2row(const vector<vector<double>>& matrix, int filter_size, int stride,
            vector<vector<double>>* matrix_im2row) {
  matrix_im2row->clear();

  int in_height = matrix.size();
  int in_width = matrix[0].size();
  for (int i_row = 0; i_row < in_height - filter_size + 1; i_row += stride) {
    for (int i_col = 0; i_col < in_width - filter_size + 1; i_col += stride) {
      // for each window
      int n_elements = filter_size * filter_size;
      vector<double> sub_mat(n_elements, 0.0);
      
      for (int f_row = 0; f_row < filter_size; ++f_row) {
        for (int f_col = 0; f_col < filter_size; ++f_col) {
          sub_mat[f_row * filter_size + f_col] = matrix[i_row + f_row][i_col + f_col];
        }
      }
      matrix_im2row->emplace_back(std::move(sub_mat));
    }
  }
}

} // namespace para