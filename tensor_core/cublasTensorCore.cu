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
#define MATRIX_M 256
#define MATRIX_N 256
#define MATRIX_K 256

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

int main(int argc, char* argv[]) {
   float *a_fp32;
   float *b_fp32;
   half *a_fp16;
   half *b_fp16;
   float *c;
   float *c_cublas;
   float *c_host_cublas;
   
   curandGenerator_t gen;
   cublasHandle_t cublasHandle;
   
   cudaEvent_t startcublas;
   cudaEvent_t stopcublas;
   
   cudaErrCheck(cudaEventCreate(&startcublas));
   cudaErrCheck(cudaEventCreate(&stopcublas));
   
   cublasErrCheck(cublasCreate(&cublasHandle));
   // Enable tensor cores
   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TF32_TENSOR_OP_MATH));
   // Disable tensor cores
   // cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_PEDANTIC_MATH));
   
   cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

   cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));

   c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

   curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

   curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
   curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));

   // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
   convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);

   // curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));
   
   // curandErrCheck(curandDestroyGenerator(gen));
   
   // cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));

   cudaErrCheck(cudaMemset(c_cublas, 0, MATRIX_M * MATRIX_N * sizeof(float)));
   float alpha = 1.0f;
   float beta = 0.0f;
   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
   
   // Now using cuBLAS
   printf("Running with cuBLAS...\n");
   float average_time = 0.0;
   int n_iter = 100;
   for(int i=0; i<n_iter; i++) {
      float elapse_time = 0.0;
      cudaErrCheck(cudaEventRecord(startcublas));
      cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                  MATRIX_M, MATRIX_N, MATRIX_K, 
                  &alpha,
                  a_fp32, CUDA_R_32F, MATRIX_M,
                  b_fp32, CUDA_R_32F, MATRIX_K,
                  &beta, 
                  c_cublas, CUDA_R_32F, MATRIX_M,
                  // CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
                  CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      // cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
      //             MATRIX_M, MATRIX_N, MATRIX_K, 
      //             &alpha,
      //             a_fp16, CUDA_R_16F, MATRIX_M,
      //             b_fp16, CUDA_R_16F, MATRIX_K,
      //             &beta, 
      //             c_cublas, CUDA_R_32F, MATRIX_M,
      //             // CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
      //             CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      cudaErrCheck(cudaEventRecord(stopcublas));
      cudaErrCheck(cudaEventSynchronize(stopcublas));
      cudaErrCheck(cudaEventElapsedTime(&elapse_time, startcublas, stopcublas));
      average_time += elapse_time;
   }
   average_time = average_time / n_iter;

   // Error checking
   cudaErrCheck(cudaMemcpy(
      c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost
      ));
   printf("cublas took %fms\n", average_time);

   cudaErrCheck(cudaEventDestroy(startcublas));             
   cudaErrCheck(cudaEventDestroy(stopcublas));
   
   cudaErrCheck(cudaFree(a_fp32));
   cudaErrCheck(cudaFree(b_fp32));
   cudaErrCheck(cudaFree(a_fp16));
   cudaErrCheck(cudaFree(b_fp16));

   cudaErrCheck(cudaFree(c));
   cudaErrCheck(cudaFree(c_cublas));
   
   free(c_host_cublas);

   cudaErrCheck(cudaDeviceReset());
   return 0;
}


