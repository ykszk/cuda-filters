#ifndef CUFILTERS_H
#define CUFILTERS_H

#ifdef _WINDOWS
#define EXPORT extern __declspec(dllexport)
#else
#define EXPORT

#endif
#ifdef __cplusplus
extern "C"{
#endif

  EXPORT void medianFilter3D(const float* input, float* output, const int *dims, int filter_size, const char* padopt);
  EXPORT void averageResample3D(const float* input, float* output, const int *dims, int resample_size);
  EXPORT void setDevice(int id);

#ifdef __cplusplus
}
#endif

#endif /* CUFILTERS_H */
