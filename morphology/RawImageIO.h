#ifndef RAWIMAGEIO_H
#define RAWIMAGEIO_H
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

template <typename T>
void load_raw(T* ptr, const std::string &filename, size_t size)
{
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);

  if (!ifs) {
    std::cerr << "Couldn't open " << filename << std::endl;
    exit(1);
  }
  ifs.read(reinterpret_cast<char*>(ptr), sizeof(T)*size);
}

template <typename T>
T* load_raw(const std::string &filename, size_t size)
{
  T* ptr = new T[size];
  load_raw(ptr, filename, size);
  return ptr;
}

template <typename T>
void save_raw(const T* ptr, size_t size, const std::string &filename)
{
  std::ofstream ofs(filename, std::ios::out | std::ios::binary);
  if (!ofs) {
    std::cerr << "Can't open " << filename << std::endl;
    exit(1);
  }
  ofs.write(reinterpret_cast<const char*>(ptr), sizeof(T)*size);
  ofs.close();
}

#endif /* RAWIMAGEIO_H */
