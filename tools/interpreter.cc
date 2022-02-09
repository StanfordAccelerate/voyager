#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <stdint.h>
#include "universal/number/posit/posit.hpp"

#define VERBOSE

using Real = sw::universal::posit<8, 1>;

void print_posit(Real posit) {
  int64_t* num = reinterpret_cast<int64_t*>(&posit);
  std::string result;
  for (int i = 0; i < 64; i++)
  {
    result = std::to_string(*num & 1) + result;
    *num = *num >> 1;
  }
  std::cout << result << std::endl;
}

int readp(const std::string &filename)
{
  // Read file into vector
  std::ifstream file(filename, std::ios::binary);
  if (!file.good())
    throw std::runtime_error("File \"" + filename + "\" does not exist");
  std::ostringstream ss;
  ss << file.rdbuf();
  const std::string &s = ss.str();
  std::vector<char> vec(s.begin(), s.end());

assert(vec.size() == 1000);
  int index = 0;
  Real mx;
  uint64_t tmp = (uint64_t)vec[0];
  memcpy(&mx, &tmp, sizeof(Real));
  print_posit(mx);

  for (int i = 0;i < vec.size();i++)
  {
    Real posit;
tmp = (uint64_t)vec[i];
    memcpy(&posit, &tmp, sizeof(Real));
    #ifdef VERBOSE
        std::cout << posit << " " << mx <<" "<< i << std::endl;
    #endif VERBOSE

      if (posit >= mx)
      {
          mx = posit;
          index = i;
      }
  }

  file.close();

std::cout << index << std::endl;
  return index;
}

int main(int argc, char **argv)
{
  // Check the number of parameters
  if (argc != 3)
  {
    // Tell the user how to run the program
    std::cerr << "usage: " << argv[0] << "infile"
              << "outfile" << std::endl;
    return 1;
  }

  std::string infile = std::string(argv[1]);
  std::string outfile = std::string(argv[2]);

  int pos = readp(infile);

  std::ofstream wf(outfile, std::ios::out | std::ios::binary);
  if (!wf.good())
    throw std::runtime_error("File \"" + outfile + "\" does not exist");

std::string postring = std::to_string(pos) + '\n';
#ifdef VERBOSE
std::cout << pos << " " <<postring << std::endl;
#endif
  wf.write(postring.c_str(), sizeof(postring.size()));
  wf.close();


    
  return 0;
}