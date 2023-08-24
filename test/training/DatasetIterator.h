#pragma once

#include <string>
#include <vector>

class DatasetIterator {
 public:
  DatasetIterator(const std::string &path);

  std::string getSample();

 private:
  int index;
  std::vector<std::string> folders;
};