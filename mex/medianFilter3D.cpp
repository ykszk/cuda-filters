#include "cuFilters.h"
#include <mex.h>
#include <algorithm>

void mexFunction(int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
  /* check for proper number of arguments */
  if(nrhs!=3) {
    mexErrMsgIdAndTxt("cuFilters:medianFilter3D:nrhs","Three inputs required.");
  }
  if(nlhs!=1) {
    mexErrMsgIdAndTxt("cuFilters:medianFilter3D:nlhs","One output required.");
  }

  // Get filter size
  if( !mxIsDouble(prhs[1]) ||
      mxIsComplex(prhs[1]) ||
      mxGetNumberOfElements(prhs[1])!=1 ) {
    mexErrMsgIdAndTxt("cuFilters:medianFilter3D:notScalar","Filter size must be a scalar.");
  }
  int filter_size = mxGetScalar(prhs[1]);
  if (mxGetNumberOfDimensions(prhs[0]) != 3) {
    mexErrMsgIdAndTxt("cuFilters:medianFilter3D:inputSize", "3D matrix input is required.");
  }
  // Get padopt
  const char* padopt = mxArrayToString(prhs[2]);
  // Get input pointer
  float *input_ptr = reinterpret_cast<float*>(mxGetPr(prhs[0]));
  const mwSize *mw_dims = mxGetDimensions(prhs[0]);
  int dims[3];
  std::copy(mw_dims, mw_dims + 3, dims);
  // Get output pointer
  size_t dims_t[3];
  std::copy(mw_dims, mw_dims + 3, dims_t);
  plhs[0] = mxCreateUninitNumericArray(3, dims_t, mxSINGLE_CLASS, mxREAL);
  float *output_ptr = reinterpret_cast<float*>(mxGetPr(plhs[0]));
  try {
    medianFilter3D(input_ptr, output_ptr, dims, filter_size, padopt);
  } catch (std::exception &e) {
    mexErrMsgIdAndTxt("cuFilters:medianFilter3D:exception", e.what());
  }
}
