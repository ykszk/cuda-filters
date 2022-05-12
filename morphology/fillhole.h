#ifndef FILLHOLE_H
#define FILLHOLE_H
#include <cstdint>
#include <cuda_runtime.h>

void fillhole2D(const uint8_t *h_input, uint8_t *h_output, const int2 &dims);
void fillhole2D_consecutive(const uint8_t * h_input_volume, uint8_t * h_output_volume, const int3 & dims3d);
#endif /* FILLHOLE_H */
