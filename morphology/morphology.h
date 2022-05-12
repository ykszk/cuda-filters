#ifndef MORPHOLOGY_H
#define MORPHOLOGY_H
#include <cuda_runtime.h>

template <typename T>
void dilate2D(const T * h_input, T * h_output, const int2 & dims, const int *h_footprint, const int2 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode);
template <typename T>
void erode2D(const T * h_input, T * h_output, const int2 & dims, const int *h_footprint, const int2 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode);
template <typename T>
void close2D(const T * h_input, T * h_output, const int2 & dims, const int *h_footprint, const int2 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode);
template <typename T>
void open2D(const T * h_input, T * h_output, const int2 & dims, const int *h_footprint, const int2 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode);

template <typename T>
void dilate3D(const T * h_input, T * h_output, const int3 & dims, const int *h_footprint, const int3 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode);
template <typename T>
void erode3D(const T * h_input, T * h_output, const int3 & dims, const int *h_footprint, const int3 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode);
template <typename T>
void close3D(const T * h_input, T * h_output, const int3 & dims, const int *h_footprint, const int3 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode);
template <typename T>
void open3D(const T * h_input, T * h_output, const int3 & dims, const int *h_footprint, const int3 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode);

// low level apis
enum class MOP {Dilation,Erosion};

template <typename T, typename ImageType>
void operate2D(MOP op, ImageType tex, T* d_output, const int2 &dims, const int *d_footprint, const int2 &footprint_dims);
template <typename T, typename ImageType>
void operate3D(MOP op, ImageType tex, T* d_output, const int3 &dims, const int *d_footprint, const int3 &footprint_dims);
#endif /* MORPHOLOGY_H */
