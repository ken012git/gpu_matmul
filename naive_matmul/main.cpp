/*
The code is adapted from https://github.com/lzhengchun/matrix-cuda
*/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <chrono>
#include <iostream>
#include "gpu_cuda_mm.hpp"
#include "cpu_mm.hpp"


float run_gpu_cuda_mm(int n_iter, int8_t *h_a, int8_t*h_b, int8_t *h_c, int m, int n, int k) {
    float average_time = GPU::launch_gpu_cuda_mm(
        n_iter, h_a, h_b, h_c, m, n, k);
    // // initial timer for each cuda thread
    // float *h_timer; // use floating point to prevent overflow
    // int timer_size = sizeof(float)*grid_rows*grid_cols*BLOCK_SIZE*BLOCK_SIZE*3;
    // h_timer = (float*) malloc(timer_size);

//     float avg_load=0, avg_compute=0, avg_store=0;
//     float num_timers = grid_rows*grid_cols*BLOCK_SIZE*BLOCK_SIZE;
//     for (int gy=0; gy < grid_rows; ++gy) {
//         for (int gx=0; gx < grid_cols; ++gx) {
//             for (int i = 0; i < BLOCK_SIZE; ++i)
//             {
//                 for (int j = 0; j < BLOCK_SIZE; ++j)
//                 {
//                     int idx = i*BLOCK_SIZE*3 + j*3;
//                     avg_load += h_timer[idx + 0];
//                     avg_compute += h_timer[idx + 1];
//                     avg_store += h_timer[idx+ 2];
//                 }
//             }
//         }
//     }
//     // A100 boost clock 1410 MHz
//     // The A100 SM includes new third-generation Tensor Cores that each perform 256 FP16/FP32 FMA operations per clock.
//     // A100 has four Tensor Cores per SM, which together deliver 1024 dense FP16/FP32 FMA operations per clock,
//     // Each SM in A100 computes a total of 64 FP64 FMA operations/clock (or 128 FP64 operations/clock)
//     avg_load = avg_load / num_timers / T;
//     avg_compute = avg_compute / num_timers / T;
//     avg_store = avg_store / num_timers / T;
//     printf("average load: %f\n", avg_load);
//     printf("average compute: %f\n", avg_compute);
//     printf("average store: %f\n", avg_store);

    return average_time;
}


float run_cpu_mm(int n_iter, int8_t *h_a, int8_t*h_b, int8_t *h_c_, int m, int n, int k) {

    // start the CPU version
    std::chrono::high_resolution_clock::time_point t1, t2;
    float total_time = 0.0;
    t1 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t<n_iter; t++) {
        // Launch 
        CPU::matrix_mult(h_a, h_b, h_c_, m, n, k);
    }
    t2 = std::chrono::high_resolution_clock::now();
    total_time += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    float average_time = total_time / n_iter;
    printf("Average time elapsed over %d iteration on matrix multiplication of %dx%d ."
        " %dx%d on CPU: %f ms.\n\n", n_iter, m, n, n, k, average_time);

    return average_time;
}


/*
*********************************************************************
function name: main
description: test and compare
parameters: 
            none
return: none
*********************************************************************
*/
int main(int argc, char const *argv[])
{
    int m, n, k;
    /* Fixed seed for illustration */
    srand(3333);
    printf("please type in m n and k\n");
    scanf("%d %d %d", &m, &n, &k);
    printf("get matrix size: %d x %d x %d \n", m, n, k);
    if(m != n || n != k)
    {
        printf("Only support squre matrix multiplication");
        return 0;
    }
    int n_iter = 10000;
    // printf("please type in number of runs \n");
    // scanf("%d", &n_iter);
    printf("test %d runs \n", n_iter);
    if(n_iter < 1)
    {
        printf("number of runs must > 1");
        return 0;
    }
    // allocate memory in host DRAM
    int8_t *h_a, *h_b, *h_c, *h_c_;
    h_a  = (int8_t*) malloc(sizeof(int8_t)*m*n);
    h_b  = (int8_t*) malloc(sizeof(int8_t)*n*k);
    h_c  = (int8_t*) malloc(sizeof(int8_t)*m*k);  // gpu result
    h_c_ = (int8_t*) malloc(sizeof(int8_t)*m*k);  // cpu result

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 127;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 127;
        }
    }

    float avg_cuda_gpu = run_gpu_cuda_mm(n_iter, h_a, h_b, h_c, m, n, k);
    // float avg_cpu = run_cpu_mm(n_iter, h_a, h_b, h_c_, m, n, k);

    // // validate results computed by GPU
    // int all_ok = 1;
    // for (int i = 0; i < m; ++i)
    // {
    //     for (int j = 0; j < k; ++j)
    //     {
    //         // printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_c_[i*k + j], i, j, h_c[i*k + j]);
    //         if(h_c_[i*k + j] != h_c[i*k + j])
    //         {
    //             all_ok = 0;
    //         }
    //     }
    //     //printf("\n");
    // }

    // // roughly compute speedup
    // if(all_ok)
    // {
    //     printf("cpu time: %.3f ms\n", avg_cpu);
    //     printf("gpu time: %.3f ms\n", avg_cuda_gpu);
    //     printf("all results are correct!!!, speedup = %.1fx\n", avg_cpu / avg_cuda_gpu);
    // }
    // else
    // {
    //     printf("incorrect results\n");
    // }

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_);
    return 0;
}