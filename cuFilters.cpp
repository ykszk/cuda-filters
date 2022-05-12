#include "cuFilters.h"
#include "medianFilter3D.h"
#include "averageResample3D.h"
#include <cstring>
#include <iostream>
using namespace std;

void medianFilter3D(const float* input, float* output, const int *dims, int filter_size, const char* padopt)
{
  cudaTextureAddressMode addressMode;
  if (strcmp(padopt, "clamp") == 0) {
    addressMode = cudaAddressModeClamp;
  }
  else if (strcmp(padopt, "mirror") == 0) {
    addressMode = cudaAddressModeMirror;
  }
  else if (strcmp(padopt, "wrap") == 0) {
    addressMode = cudaAddressModeWrap;
  }
  else if (strcmp(padopt, "border") == 0) {
    addressMode = cudaAddressModeBorder;
  }
  else {
    throw std::invalid_argument("Unsupported padopt.");
  }
  medianFilter3D(input, output, make_int3(dims[0], dims[1], dims[2]), filter_size, addressMode);
}

void averageResample3D(const float * input, float * output, const int * dims, int resample_size)
{
  averageResample3D(input, output, make_int3(dims[0], dims[1], dims[2]), resample_size);
}

void setDevice(int id)
{
  cudaSetDevice(id);
}
