// MIT License
//
// Copyright (c) 2020 xiw
// \author wang xi
// Word count program using MPI API.


#include "word_count.h"

namespace para {

void WordCounter::WordCount(int argc, char* argv[]) {
  if (argc != 4) {
    ::printf("Usage: %s -f in_file out_file\nor: %s -d in_dir out_file\n", argv[0], argv[0]);
    return;
  }
  
  ::MPI_Init(&argc, &argv);
  if (::strcmp("-f", argv[1]) == 0) {
    string in_file(argv[2]);
    string out_file(argv[3]);

    CountLargeFile(in_file, out_file);
  } else if (::strcmp("-d", argv[1]) == 0) {
    string in_dir(argv[2]);
    string out_file(argv[3]);
    
    CountSmallFiles(in_dir, out_file);
  }
  ::MPI_Finalize();
}

void WordCounter::CountSmallFiles(const string& in_dir, const string& out_file) {
  int id = -1;
  ::MPI_Comm_rank(MPI_COMM_WORLD, &id);
  int processes_size = -1;
  ::MPI_Comm_size(MPI_COMM_WORLD, &processes_size);

  struct dirent *ptr;      
  auto dir = opendir(in_dir.c_str());

  vector<string> files;  
  while( (ptr=::readdir(dir)) != NULL){  
    if(ptr->d_name[0] == '.')  
        continue; 
    string filename(in_dir);
    filename.append("/");
    filename.append(ptr->d_name);

    files.push_back(filename);  
  }  
  ::closedir(dir);

  int single_size = files.size() / processes_size + 1;
  int start_num = single_size * id;
  int end_num = single_size * (id+1) > files.size() ? files.size() : single_size * (id+1);

  if (0 == id) {
    // master collect the results
    MasterSmall(files, start_num, end_num, out_file, processes_size, MPI_COMM_WORLD);
  } else {
    WorkerSmall(files, start_num, end_num, 0, MPI_COMM_WORLD);
  }
}

void WordCounter::CountLargeFile(const string& in_file, const string& out_file) {
  int id = -1;
  ::MPI_Comm_rank(MPI_COMM_WORLD, &id);
  int processes_size = -1;
  ::MPI_Comm_size(MPI_COMM_WORLD, &processes_size);

  struct stat statbuf;
  stat(in_file.c_str(), &statbuf);
  
  int total_size = statbuf.st_size;

  int single_size = total_size / processes_size + 1;
  int start_pos = single_size * id;

  if (0 == id) {
    // master collect the results
    //printf("id = %d\n", id);
    MasterLarge(in_file, single_size, out_file, processes_size, MPI_COMM_WORLD);
  } else {
    //printf("id = %d\n", id);
    WorkerLarge(in_file, single_size, id, 0, MPI_COMM_WORLD);
  }
}

void WordCounter::WorkerLarge(const string& in_file, int single_size, int id, int master_id, const ::MPI_Comm& comm) {
  vector<char> text_buf(single_size+2, 0);

  int start_pos = single_size * id;
  auto f = ::fopen(in_file.c_str(), "r");
  ::fseek(f, start_pos-1, 0);
  
  ::fread(text_buf.data(), sizeof(char), single_size+1, f);

  char end_ch = text_buf.back();
  if (IsAlpha(end_ch)) {
    ::fread(&end_ch, sizeof(char), 1, f);
    while (IsAlpha(end_ch)) {
      text_buf.emplace_back(end_ch);
      ::fread(&end_ch, sizeof(char), 1, f);
    }
  }
  ::fclose(f);
  //printf("worker:\n%s\n", text_buf.data());

  std::unordered_map<string, int> word_dict;
  ProcessText(text_buf, &word_dict);

  vector<char> words;
  vector<int> counts;
  Map2Vec(word_dict, &words, &counts);

  // send response back to master
  ::MPI_Send(words.data(), words.size(), MPI_CHAR, master_id, PUT_RESULT, comm);
  ::MPI_Send(counts.data(), counts.size(), MPI_INT, master_id, PUT_RESULT, comm);
}

void WordCounter::MasterLarge(const string& in_file, int single_size, const string& out_file, int num_processes, const ::MPI_Comm comm) {
  vector<char> text_buf(single_size+1, 0);
  text_buf[0] = ' ';

  int start_pos = 0;
  auto f = ::fopen(in_file.c_str(), "r");
  ::fseek(f, 0, 0);
  
  ::fread(text_buf.data()+1, sizeof(char), single_size, f);
  
  char end_ch = text_buf.back();
  if (IsAlpha(end_ch)) {
    ::fread(&end_ch, sizeof(char), 1, f);
    while (IsAlpha(end_ch)) {
      text_buf.emplace_back(end_ch);
      ::fread(&end_ch, sizeof(char), 1, f);
    }
  }
  
  fclose(f);
  //printf("master:\n%s\n", text_buf.data());

  std::unordered_map<string, int> word_dict;
  ProcessText(text_buf, &word_dict);

  // receive response
  int terminated_processes = 0;
  while (num_processes - 1 > terminated_processes) {
    ::MPI_Status status;

    ::MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
    int count = -1;
    MPI_Get_count(&status, MPI_CHAR, &count);
    vector<char> words(count+1, 0);
    ::MPI_Recv(words.data(), count, MPI_CHAR, status.MPI_SOURCE, status.MPI_TAG, comm, &status);

    ::MPI_Probe(status.MPI_SOURCE, status.MPI_TAG, comm, &status);
    MPI_Get_count(&status, MPI_INT, &count);
    vector<int> counts(count, 0);

    ::MPI_Recv(counts.data(), count, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, comm, &status);
    int j = 0;
    for(int i = 0; i < counts.size(); ++i){
      string word(words.data() + j);
      
      if (word_dict.find(word) == word_dict.end()) {
        word_dict[word] = 0;
      }
      word_dict[word] += counts[i];

      while(j < words.size() && 0 != words[j]){
        ++j;
      }
      ++j;
    }
    terminated_processes++;
  }

  f = ::fopen(out_file.c_str(), "w");
  for(auto i : word_dict){
    string s(i.first);
    s.append(" ");
    char count_buf[10] = {0};
    sprintf(count_buf, "%d", i.second);
    s.append(count_buf);
    s.append("\n");

    ::fwrite(s.c_str(), sizeof(char), s.size(), f);
  }
  ::fclose(f);
}


void WordCounter::WorkerSmall(const vector<string> files, int start_file_num, int end_file_num, int master_id, const ::MPI_Comm& comm) {
  std::unordered_map<string, int> word_dict;
  for (int i = start_file_num; i < end_file_num; ++i) {
    struct stat statbuf;
    stat(files[i].c_str(), &statbuf);
    int total_size = statbuf.st_size;

    vector<char> text_buf(total_size+2, 0);
    text_buf[0] = ' ';

    auto f = ::fopen(files[i].c_str(), "r");
    
    ::fread(text_buf.data() + 1, sizeof(char), total_size, f);
    ::fclose(f);
    printf("worker:\n%s\n", text_buf.data());
    ProcessText(text_buf, &word_dict);
  }

  vector<char> words;
  vector<int> counts;
  
  Map2Vec(word_dict, &words, &counts);
  for(auto i : counts){
    printf("%d\n", i);
  }
  // send response back to master
  ::MPI_Send(words.data(), words.size(), MPI_CHAR, master_id, PUT_RESULT, comm);
  ::MPI_Send(counts.data(), counts.size(), MPI_INT, master_id, PUT_RESULT, comm);  
}

void WordCounter::MasterSmall(const vector<string> files, int start_file_num, int end_file_num, const string& out_file, int num_processes, const ::MPI_Comm comm){
  std::unordered_map<string, int> word_dict;
  for (int i = start_file_num; i < end_file_num; ++i) {
    struct stat statbuf;
    stat(files[i].c_str(), &statbuf);
    int total_size = statbuf.st_size;

    vector<char> text_buf(total_size+2, 0);
    text_buf[0] = ' ';

    auto f = ::fopen(files[i].c_str(), "r");
    
    ::fread(text_buf.data() + 1, sizeof(char), total_size, f);
    ::fclose(f);

    printf("master:\n%s\n", text_buf.data());

    ProcessText(text_buf, &word_dict);
  }


  // receive response
  int terminated_processes = 0;
  while (num_processes-1 > terminated_processes){
    ::MPI_Status status;
    //printf("start to receive\n");
    ::MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
    int count = -1;
    MPI_Get_count(&status, MPI_CHAR, &count);
    vector<char> words(count+1, 0);
    ::MPI_Recv(words.data(), count, MPI_CHAR, status.MPI_SOURCE, status.MPI_TAG, comm, &status);
    //printf("succeeded receiving\n");

    ::MPI_Probe(status.MPI_SOURCE, status.MPI_TAG, comm, &status);
    MPI_Get_count(&status, MPI_INT, &count);
    vector<int> counts(count, 0);

    ::MPI_Recv(counts.data(), count, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, comm, &status);
    int j = 0;
    for(int i = 0; i < counts.size(); ++i){
      string word(words.data() + j);
      if (word_dict.find(word) == word_dict.end()) {
        word_dict[word] = 0;
      }
      word_dict[word] += counts[i];

      while(j < words.size() && 0 != words[j]){
        ++j;
      }
      ++j;
    }
    terminated_processes++;
  } 

  auto f = ::fopen(out_file.c_str(), "w");
  for(auto i : word_dict){
    string s(i.first);
    s.append(" ");
    char count_buf[10] = {0};
    sprintf(count_buf, "%d", i.second);
    s.append(count_buf);
    s.append("\n");

    ::fwrite(s.c_str(), sizeof(char), s.size(), f);
  }
  ::fclose(f);
}

inline bool WordCounter::IsAlpha(const char& c) {
  if (c == '\'' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
    return true;
  }
  return false;
}

void WordCounter::ProcessText(const vector<char>& buf, std::unordered_map<string, int>* word_dict) {
  int iter = 0;
  while ( iter < buf.size() && IsAlpha(buf[iter]) ) {
    ++iter;
  }

  for (; iter < buf.size(); ++iter) {
    while ( iter < buf.size() && !IsAlpha(buf[iter]) ) {
      ++iter;
    }
    if (iter == buf.size()){
      break;
    }

    auto iter_start = iter;
    while ( iter < buf.size() && IsAlpha(buf[iter]) ) {
      ++iter;
    }
    string word(buf.data() + iter_start, iter - iter_start);
    //printf("%s\n", word.c_str());

    if (word_dict->find(word) == word_dict->end()) {
      word_dict->emplace(std::make_pair(word, 0));
    }
    (*word_dict)[word] += 1;
  }
}

void WordCounter::Map2Vec(const std::unordered_map<string, int>& word_dict, vector<char>* words, vector<int>* counts) {
  for (auto i : word_dict) {
    for(auto ch : i.first) {
      words->emplace_back(ch);
    }
    words->emplace_back('\0');

    counts->emplace_back(i.second);
  }
}

} // namespace para
