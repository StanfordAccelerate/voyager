#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "universal/number/posit/posit.hpp"

using Real = sw::universal::posit<8, 1>;

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

  // check:
//   std::copy(vec.begin(), vec.end(), std::ostream_iterator<char>(std::cout));
//   buf = new double[vec.size() / 8];
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
    // std::cout << size << std::endl;
    Real intermediate = tmp[i];
    char *posit = reinterpret_cast<char *>(&intermediate);
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