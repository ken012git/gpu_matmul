#pragma once
// Max threads / SM 2048
// Max thread blocks / SM 32
// Shared Memory Size / SM	164 KB
#define BLOCK_SIZE 32 // 32x32 = 1024 

#include <stdint.h>

extern "C" 
{
namespace GPU
{
    // __global__ void matrix_mult(
    //     int *d_a, int *d_b, int *d_result, int n);

    // __global__ void matrix_mult_timer(
    //     int *d_a, int *d_b, int *d_result, int n, float* d_timer);

    float launch_gpu_cuda_mm(int n_iter, int8_t *h_a, int8_t*h_b, int8_t *h_c, int m, int n, int k);

    // float launch_gpu_cuda_mm_timer(int n_iter, int8_t *h_a, int8_t*h_b, int8_t *h_c,
    //     float *h_timer, int m, int n, int k, int timer_size);
}
}