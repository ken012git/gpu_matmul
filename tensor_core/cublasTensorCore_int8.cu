#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}

#include <mma.h>
using namespace nvcuda;

// Must be multiples of 16 for wmma code to work
#define MATRIX_M 1024
#define MATRIX_N 1024
#define MATRIX_K 1024

void random_init(int8_t* a, int m, int n) {
    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = (int8_t) rand() % 127;
        }
    }
}



int main(int argc, char* argv[]) {
   int8_t *a_int8_host, *a_int8_device;
   int8_t *b_int8_host, *b_int8_device;
   int32_t *c_int32_host, *c_int32_device;
   printf("sizeof(int8_t): %lu, sizeof(ing32_t): %lu", sizeof(int8_t), sizeof(int32_t));

   a_int8_host  = (int8_t*) malloc(sizeof(int8_t) * MATRIX_M * MATRIX_K);
   b_int8_host  = (int8_t*) malloc(sizeof(int8_t) * MATRIX_K * MATRIX_N);
   c_int32_host  = (int32_t*) malloc(sizeof(int32_t) * MATRIX_M * MATRIX_N);
   random_init(a_int8_host, MATRIX_M, MATRIX_K);
   random_init(b_int8_host, MATRIX_K, MATRIX_N);

   cublasHandle_t cublasHandle;
   cudaEvent_t startcublas;
   cudaEvent_t stopcublas;
   
   cudaErrCheck(cudaEventCreate(&startcublas));
   cudaErrCheck(cudaEventCreate(&stopcublas));
   cublasErrCheck(cublasCreate(&cublasHandle));
   // Enable tensor cores
   // cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));
   // Disable tensor cores
   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_PEDANTIC_MATH));
   
   cudaErrCheck(cudaMalloc((void**)&a_int8_device, MATRIX_M * MATRIX_K * sizeof(int8_t)));
   cudaErrCheck(cudaMalloc((void**)&b_int8_device, MATRIX_K * MATRIX_N * sizeof(int8_t)));
   cudaErrCheck(cudaMalloc((void**)&c_int32_device, MATRIX_M * MATRIX_N * sizeof(int32_t)));

   cudaErrCheck(cudaMemcpy(a_int8_device, a_int8_host,
      MATRIX_M * MATRIX_K * sizeof(int8_t), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(b_int8_device, b_int8_host,
      MATRIX_K * MATRIX_N * sizeof(int8_t), cudaMemcpyHostToDevice));

   int8_t alpha = 1;
   int8_t beta = 0;
   printf("\nM = %d, N = %d, K = %d. alpha = %d, beta = %d\n\n",
      MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
   
   // Now using cuBLAS
   printf("Running with cuBLAS...\n");
   float average_time = 0.0;
   int n_iter = 10000;
   for(int i=0; i<n_iter; i++) {
      float elapse_time = 0.0;
      cudaErrCheck(cudaMemset(c_int32_device, 0, MATRIX_M * MATRIX_N * sizeof(int32_t)));
      cudaErrCheck(cudaEventRecord(startcublas));
      cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
                  MATRIX_M, MATRIX_N, MATRIX_K, 
                  &alpha,
                  a_int8_device, CUDA_R_8I, MATRIX_M,
                  b_int8_device, CUDA_R_8I, MATRIX_K,
                  &beta, 
                  c_int32_device, CUDA_R_32I, MATRIX_M,
                  // CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));  // Enable Tensor Cores
                  CUBLAS_COMPUTE_32I_PEDANTIC, CUBLAS_GEMM_DEFAULT));  // Disable Tensor Cores
      cudaErrCheck(cudaEventRecord(stopcublas));
      cudaErrCheck(cudaEventSynchronize(stopcublas));
      cudaErrCheck(cudaEventElapsedTime(&elapse_time, startcublas, stopcublas));
      average_time += elapse_time;
   }
   average_time = average_time / n_iter;
   // Error checking
   // cudaErrCheck(
   //    cudaMemcpy(
   //       c_int32_host, c_int32_device,
   //       MATRIX_M * MATRIX_N * sizeof(int32_t),
   //       cudaMemcpyDeviceToHost
   //    )
   // );
   printf("cublas took %f ms\n", average_time);

   cudaErrCheck(cudaEventDestroy(startcublas));             
   cudaErrCheck(cudaEventDestroy(stopcublas));
   
   cudaErrCheck(cudaFree(a_int8_device));
   cudaErrCheck(cudaFree(b_int8_device));
   cudaErrCheck(cudaFree(c_int32_device));
   
   free(a_int8_host);
   free(b_int8_host);
   free(c_int32_host);
   cudaErrCheck(cudaDeviceReset());
   return 0;
}


