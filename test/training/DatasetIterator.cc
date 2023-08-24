#include "DatasetIterator.h"

#include <filesystem>

DatasetIterator::DatasetIterator(const std::string& path) : index(0) {
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    // get folder name
    std::string folderName = entry.path().filename().string();

    // skip params folder
    if (folderName == "params") {
      continue;
    }

    folders.push_back(entry.path().string());
  }
}

std::string DatasetIterator::getSample() {
  if (index >= folders.size()) {
    index = 0;
  }

  std::string folder = folders[index];
  index++;
  return folder;
}
