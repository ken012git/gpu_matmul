#include "gpu_cuda_mm.hpp" // header in local directory

#include <stdio.h>
#include <stdint.h>
/*
The code is adapted from https://github.com/lzhengchun/matrix-cuda
*/
/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *  
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *  
 */

// Max threads / SM 2048
// Max thread blocks / SM 32
// Shared Memory Size / SM	164 KB
// #define BLOCK_SIZE 4 // 32x32 = 1024 

/*
*********************************************************************
function name: gpu_square_matrix_mult
description: dot product of two matrix (not only square) in GPU
parameters: 
            &a GPU device pointer to a n X n matrix (A)
            &b GPU device pointer to a n X n matrix (B)
            &c GPU device output purpose pointer to a n X n matrix (C) 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
return: none
*********************************************************************
*/
__global__ void matrix_mult(int8_t *d_a, int8_t *d_b, int8_t *d_result, int n) 
{
    __shared__ int8_t tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int8_t tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }  
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}


__global__ void matrix_mult_timer(int8_t *d_a, int8_t *d_b, int8_t *d_result, int n, float* d_timer) 
{
    __shared__ int8_t tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int8_t tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    int t_idx = blockIdx.y * (gridDim.x*blockDim.x*blockDim.y*3) +
        blockIdx.x * (blockDim.x*blockDim.y*3) +
        threadIdx.y * (blockDim.x*3) + threadIdx.x*3;


    clock_t start_time, stop_time;
    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        start_time = clock(); 
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }  
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        stop_time = clock();
        d_timer[t_idx + 0] += (float)(stop_time - start_time);
        __syncthreads();

        start_time = clock(); 
        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        stop_time = clock();
        d_timer[t_idx + 1] += (float)(stop_time - start_time);
        __syncthreads();
    }
    if(row < n && col < n)
    {
        start_time = clock(); 
        d_result[row * n + col] = tmp;
        stop_time = clock();
        d_timer[t_idx + 2] += (float)(stop_time - start_time);
    }
}

extern "C" 
float launch_gpu_cuda_mm(int n_iter, int8_t *h_a, int8_t*h_b, int8_t *h_c, int m, int n, int k) {
   // Allocate memory space on the device 
    int8_t *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int8_t)*m*n);
    cudaMalloc((void **) &d_b, sizeof(int8_t)*n*k);
    cudaMalloc((void **) &d_c, sizeof(int8_t)*m*k);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int8_t)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int8_t)*n*k, cudaMemcpyHostToDevice);

    // get device grid and block
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    printf("Device grid size: %d x %d \n", grid_rows, grid_cols);
    printf("Device block size: %d x %d \n", BLOCK_SIZE, BLOCK_SIZE);

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    float elapsed_time, total_time, average_time;
    for (int t = 0; t<n_iter; t++) {
        cudaEventRecord(start);  // Start timer 
        matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);  // Launch kernel
        cudaEventRecord(stop);  // End timer 
        cudaMemcpy(h_c, d_c, sizeof(int8_t)*m*k, cudaMemcpyDeviceToHost);
        // time counting terminate
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        total_time += elapsed_time;
    }
    average_time = total_time / n_iter;
    printf("Average time elapsed over %d iteration on matrix multiplication of %dx%d ."
        " %dx%d on GPU: %f ms.\n\n", n_iter, m, n, n, k, average_time);
    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return average_time;
}


// profiling load, compute, store
extern "C" 
float launch_gpu_cuda_mm_timer(int n_iter, int8_t *h_a, int8_t*h_b, int8_t *h_c,
    float *h_timer, int m, int n, int k, int timer_size) {
   // Allocate memory space on the device 
    int8_t *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int8_t)*m*n);
    cudaMalloc((void **) &d_b, sizeof(int8_t)*n*k);
    cudaMalloc((void **) &d_c, sizeof(int8_t)*m*k);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int8_t)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int8_t)*n*k, cudaMemcpyHostToDevice);

    // get device grid and block
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    printf("Device grid size: %d x %d \n", grid_rows, grid_cols);
    printf("Device block size: %d x %d \n", BLOCK_SIZE, BLOCK_SIZE);

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // initial timer for each cuda thread
    float *d_timer; // use floating point to prevent overflow
    cudaMalloc((void **) &d_timer, timer_size);
    cudaMemset(d_timer, 0, timer_size); // set timer to zero

    // start to count execution time of GPU version
    float elapsed_time, total_time, average_time;
    for (int t = 0; t<n_iter; t++) {
        elapsed_time = 0;
        cudaEventRecord(start);  // Start timer 
        matrix_mult_timer<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n, d_timer);  // Launch kernel
        cudaEventRecord(stop);  // End timer 
        cudaMemcpy(h_c, d_c, sizeof(int8_t)*m*k, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_timer, d_timer, timer_size, cudaMemcpyDeviceToHost);
        // time counting terminate
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop); // Time between start and end in ms
        total_time += elapsed_time;
    }
    average_time = total_time / n_iter;
    printf("Average time elapsed over %d iteration on matrix multiplication of %dx%d ."
        " %dx%d on GPU: %f ms.\n\n", n_iter, m, n, n, k, average_time);

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_timer);
    return average_time;
}