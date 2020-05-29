// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi
// PageRank

#include "page_rank.h"

namespace para {

PageRanker::PageRanker(double damping_factor_, int max_iter_, double precision_)
    : damping_factor(damping_factor_), max_iter(max_iter_), precision(precision_) {}


void PageRanker::PageRank(const vector<vector<int>>& connections, vector<double>* page_rank) {
  vector<int> csr_indptr, csr_indices;
  vector<double> csr_data;

  Connections2CSRMatrix(connections, &csr_indptr, &csr_indices, &csr_data);

  PowerMethodPR(csr_indptr, csr_indices, csr_data, page_rank);
}


void PageRanker::Connections2CSRMatrix(const vector<vector<int>>& connections, vector<int>* csr_indptr,
                                       vector<int>* csr_indices, vector<double>* csr_data) {
  vector<vector<int>> conns {connections};

  auto less_than_ = [](const vector<int>& v1, const vector<int>& v2) -> bool {
    if (v1[0] > v2[0])
      return false;
    else if (v1[0] == v2[0])
      return v1[1] < v2[1];
    else
      return true;
  };

  std::sort(conns.begin(), conns.end(), less_than_);

  int max_node_id = conns.back()[0];

  vector<int> nodes_outs(conns.back()[0]+1, 0);
  for (const auto& v : conns) {
    ++nodes_outs[v[0]];
  }

  auto less_than_by_2_ = [](const vector<int>& v1, const vector<int>& v2) -> bool {
    if (v1[1] > v2[1])
      return false;
    else if (v1[1] == v2[1])
      return v1[0] < v2[0];
    else
      return true;
  };
  
  std::sort(conns.begin(), conns.end(), less_than_by_2_);
  max_node_id = max_node_id > conns.back()[1] ? max_node_id : conns.back()[1];
  
  csr_indptr->clear();
  csr_indices->clear();
  csr_data->clear();

  int node_id = 0;
  int num_nonzeros = 0;
  csr_indptr->emplace_back(num_nonzeros);

  for (int i = 0; i < conns.size();) {
    // no incidence to node_id
    while (conns[i][1] > node_id) {
      csr_indptr->emplace_back(num_nonzeros);
      ++node_id;
    }

    // consume all nodes with incidence to node_id
    while(i < conns.size() && conns[i][1] == node_id) {
      ++num_nonzeros;
      csr_indices->emplace_back(conns[i][0]);
      csr_data->emplace_back(1.0 / nodes_outs[conns[i][0]]);
      ++i;
    }
    csr_indptr->emplace_back(num_nonzeros);
    ++node_id;
  }

  // no incidence to node_id
  while (max_node_id >= node_id) {
    csr_indptr->emplace_back(num_nonzeros);
    ++node_id;
  }

}

void PageRanker::PowerMethodPR(const vector<int>& csr_indptr, const vector<int>& csr_indices,
                               const vector<double>& csr_data, vector<double>* page_rank) {
  int num_nodes = csr_indptr.size() - 1;

  // init vector
  vector<double> u(num_nodes, 1.0 / num_nodes);
  
  // records the position of nodes that have no out links
  vector<int> nodes_without_outlinks;
  {  
    vector<bool> nodes_(num_nodes, false); 
    for (int i : csr_indices){
      nodes_[i] = true;
    }
    for (int i = 0; i < nodes_.size(); ++i) {
      if (nodes_[i] == false) {
        nodes_without_outlinks.emplace_back(i);
      }
    }
  }
  
  vector<double> v {u};

  double max_change = 1.0;
  for (int iter = 0; iter < max_iter && max_change > precision; ++iter) {
    double sum_pr_without_outlinks = 0.0;
    for (int i : nodes_without_outlinks) {
      sum_pr_without_outlinks += u[i];
    }

    // do the matrix-vector multiplication
    //double sum_pr = 0.0;
    #pragma omp parallel for
    for (int i = 0; i < num_nodes; ++i) {
      int start = csr_indptr[i];
      int length = csr_indptr[i+1] - csr_indptr[i];

      double sum_j = 0.0;
      for(int j = csr_indptr[i]; j < csr_indptr[i+1]; ++j) {
        sum_j += csr_data[j] * u[csr_indices[j]];
      }
      sum_j += sum_pr_without_outlinks / num_nodes; 
      v[i] = sum_j * damping_factor + (1.0 - damping_factor) / num_nodes;
      //#pragma omp atomic
      //sum_pr += v[i];
    }

    // from u to v
    max_change = 0.0;
    #pragma omp parallel for
    for (int i = 0; i < num_nodes; ++i) {
      //v[i] /= sum_pr;

      double change = u[i] - v[i] > 0.0 ? u[i] - v[i] : v[i] - u[i];
      #pragma omp atomic
      max_change = max_change > change ? max_change : change;
    }

    // swap
    u.swap(v);
  }

  // swap memory
  page_rank->swap(u);
}

} // namespace para
