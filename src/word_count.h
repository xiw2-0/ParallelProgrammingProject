// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi
// Word count program using MPI API.

#ifndef WORD_COUNT_H_
#define WORD_COUNT_H_

#include <string>
#include <cstring>

#include <cstdlib>
#include <cstdio>
#include <fcntl.h>
#include <sys/unistd.h>
#include <sys/stat.h>
#include <dirent.h>

#include <vector>
#include <unordered_map>

#include <mpi.h>

using std::string;
using std::vector;

namespace para {

class WordCounter {
 public:

  WordCounter() = default;
  ~WordCounter() = default;

  // \brief Call CountLargeFile() or CountSmallFiles() according to the parameter.
  void WordCount(int argc, char* argv[]);

 private:

  // \brief Perform word counting using MPI. Scan the files in the input
  // directory and extract words from these files. Finally, output the 
  // word and counts in a given file descriptor.
  //
  // Call MPI_Init() before this method and call MPI_Finalize() after.
  //
  // \param in_dir input directory which contains a lot of small files.
  // \param out_file output file
  // \return void
  void CountSmallFiles(const string& in_dir, const string& out_file);

  // \brief Perform word counting using MPI. Scan the given file
  // and extract words from these files. Finally, output the 
  // word and counts in a given file descriptor.
  //
  // Call MPI_Init() before this method and call MPI_Finalize() after.
  //
  // \param in_file input big file to be scanned.
  // \param out_file output file
  // \return void
  void CountLargeFile(const string& in_file, const string& out_file);



  void WorkerLarge(const string& in_file, int single_size, int id, int master_id, const ::MPI_Comm& comm);

  void MasterLarge(const string& in_file, int single_size, const string& out_file, int num_processes, const ::MPI_Comm comm);


  void WorkerSmall(const vector<string> files, int start_file_num, int end_file_num, int master_id, const ::MPI_Comm& comm);

  void MasterSmall(const vector<string> files, int start_file_num, int end_file_num, const string& out_file, int num_processes, const ::MPI_Comm comm);


  void ProcessText(const vector<char>& buf, std::unordered_map<string, int>* word_dict);

  void Map2Vec(const std::unordered_map<string, int>& word_dict, vector<char>* words, vector<int>* counts);

  inline bool IsAlpha(const char& c);

 private:

  const int GET_WORK = 10101;
  const int PUT_RESULT = 10102;
};








} // namespace para



#endif