#include "medianFilter3D.h"
#include <iostream>
#include <vector>
#include "RawImageIO.h"
#include <cuda_runtime.h>
#include "timer.h"

using namespace std;
int main(int argc, char *argv[])
{
  if (argc < 7) {
    cerr << "Usage:" << argv[0] << " <input> <output> <dims.x> <dims.y> <dims.z> <filter size>" << endl;
    return 1;
  }
  int dx = atoi(argv[3]);
  int dy = atoi(argv[4]);
  int dz = atoi(argv[5]);
  int filter_size = atoi(argv[6]);
  cout << dx << "x" << dy << "x" << dz << endl;
  cout << "filter size " << filter_size << endl;
  int3 dims = make_int3(dx,dy,dz);
  int image_size = dims.x*dims.y*dims.z;
  {
    cout << "initialize device " << flush;
    timer<> t(true);
    cudaSetDevice(0);
  }
  float *input;
  input = load_raw<float>(argv[1], image_size);
  float *output;
  cudaMallocManaged(&output, image_size * sizeof(float));
  cout << "median filter" << endl;
  {
    timer<> t(true);
    medianFilter3D(input, output, dims, filter_size, cudaAddressModeMirror);
  }
  save_raw(output, image_size, argv[2]);
  return 0;
}
