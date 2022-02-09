#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "universal/number/posit/posit.hpp"

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

void print_char(char c) {
  std::string result;
  for (int i = 0; i < 8; i++)
  {
    result = std::to_string(c & 1) + result;
    c = c>> 1;
  }
  std::cout << result << std::endl;
}

size_t readdp(const std::string &filename, double *buf)
{
  // Read file into vector
  std::ifstream file(filename, std::ios::binary);
  if (!file.good())
    throw std::runtime_error("File \"" + filename + "\" does not exist");
  std::ostringstream ss;
  ss << file.rdbuf();
  const std::string &s = ss.str();
  std::vector<char> vec(s.begin(), s.end());

  memcpy(buf, vec.data(), vec.size());
  file.close();

  return vec.size() / 8;
}

void rewrite_data(std::string infile, std::string outfile)
{
  double *tmp = new double[1024*1024*12];
  size_t size = readdp(infile, tmp);
  char *buf = new char[size];
  for (size_t i = 0; i < size; i++)
  {
    // Posit conversion from double
    Real intermediate = tmp[i];
    // print_char(posit_char(&intermediate));
    // print_posit(&intermediate);
    char *posit = reinterpret_cast<char *>(&intermediate);
    // print_char(*posit);
    // // for (int i = 0; i< 8; i++)
    // // {

    // // print_char(*(posit + i));
    // // }
    buf[i] = *posit;
  }
  std::ofstream wf(outfile, std::ios::out | std::ios::binary);
  if (!wf.good())
    throw std::runtime_error("File \"" + outfile + "\" does not exist");
  wf.write(buf, size);
  wf.close();
  delete[] tmp;
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

  rewrite_data(std::string(argv[1]), std::string(argv[2]));
  return 0;
}