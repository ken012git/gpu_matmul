/**
The code is adapted from Nvidia repository.
https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSELt/matmul/matmul_example.cpp

All right reserved by Nvidia
**/
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header
#include <cublas_v2.h>        // cublas header
#include <cstdio>             // printf
#include <cstdlib>            // std::rand
#include <random>




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

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

constexpr int EXIT_UNSUPPORTED = 2;
// profiling parameters
float sparse_ratio = 0.5;
int n_iter = 1000;

// Host problem definition, row-major order
constexpr int m     = 1024; // bigger sizes may require dynamic allocations
constexpr int n     = 1024; // bigger sizes may require dynamic allocations
constexpr int k     = 1024; // bigger sizes may require dynamic allocations
auto          order = CUSPARSE_ORDER_ROW;
auto          opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
auto          opB   = CUSPARSE_OPERATION_NON_TRANSPOSE;

#if defined(F32)
    typedef float A_DTYPE;
    typedef float B_DTYPE;
    typedef float C_DTYPE;
    auto          A_device_dtype  = CUDA_R_32F;
    auto          B_device_dtype  = CUDA_R_32F;
    auto          C_device_dtype  = CUDA_R_32F;
    auto          compute_type_cublas    = CUBLAS_COMPUTE_32F_PEDANTIC;   // for cublas
    auto          compute_type_cublas_tc = CUBLAS_COMPUTE_32F_FAST_TF32;  // for cublas with tensorcore
    auto          compute_type_sparse = CUSPARSE_COMPUTE_TF32_FAST;
    unsigned alignment      = 32;
#elif defined(F16)
    typedef __half A_DTYPE;
    typedef __half B_DTYPE;
    typedef __half C_DTYPE;
    auto          A_device_dtype  = CUDA_R_16F;
    auto          B_device_dtype  = CUDA_R_16F;
    auto          C_device_dtype  = CUDA_R_16F;
    auto          compute_type_cublas    = CUBLAS_COMPUTE_16F_PEDANTIC;  // for cublas
    auto          compute_type_cublas_tc = CUBLAS_COMPUTE_16F;           // for cublas with tensorcore
    auto          compute_type_sparse = CUSPARSE_COMPUTE_16F;
    unsigned alignment      = 16;
#elif defined(I8)
    typedef int8_t  A_DTYPE;
    typedef int8_t  B_DTYPE;
    typedef int32_t C_DTYPE;
    auto          A_device_dtype  = CUDA_R_8I;
    auto          B_device_dtype  = CUDA_R_8I;
    auto          C_device_dtype  = CUDA_R_32I;
    auto          compute_type_cublas    = CUBLAS_COMPUTE_32I_PEDANTIC;  // for cublas
    auto          compute_type_cublas_tc = CUBLAS_COMPUTE_32I;           // for cublas with tensorcore
    auto          compute_type_sparse = CUSPARSE_COMPUTE_32I;
    unsigned alignment      = 8;  // cusparseLtStructuredDescriptorInit(): unsupported alignment value (8)
#else
    #warning "Undefine data type. Use float instead."
    typedef float A_DTYPE;
    typedef float B_DTYPE;
    typedef float C_DTYPE;
    auto          A_device_dtype  = CUDA_R_32F;
    auto          B_device_dtype  = CUDA_R_32F;
    auto          C_device_dtype  = CUDA_R_32F;
    auto          compute_type_cublas    = CUBLAS_COMPUTE_32F_PEDANTIC;   // for cublas
    auto          compute_type_cublas_tc = CUBLAS_COMPUTE_32F_FAST_TF32;  // for cublas with tensorcore
    auto          compute_type_sparse = CUSPARSE_COMPUTE_TF32_FAST;
    unsigned alignment      = 32;
#endif


bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
auto     num_A_rows     = (isA_transposed) ? k : m;
auto     num_A_cols     = (isA_transposed) ? m : k;
auto     num_B_rows     = (isB_transposed) ? n : k;
auto     num_B_cols     = (isB_transposed) ? k : n;
auto     num_C_rows     = m;
auto     num_C_cols     = n;
auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;
auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;
auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
auto     A_size         = A_height * lda * sizeof(A_DTYPE);
auto     B_size         = B_height * ldb * sizeof(B_DTYPE);
auto     C_size         = C_height * ldc * sizeof(C_DTYPE);


float profile_sparse_matmul(
    int n_iter, A_DTYPE * hA, B_DTYPE *hB, C_DTYPE *hC, float alpha, float beta, bool check_result, int verbose = 0) {

    float elapse_time = 0;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaErrCheck(cudaEventCreate(&start));
    cudaErrCheck(cudaEventCreate(&stop));
    //--------------------------------------------------------------------------
    // Device memory management
    A_DTYPE *dA, *dA_compressed;
    B_DTYPE *dB;
    C_DTYPE *dC, *dD;
    int    *d_valid;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(int)) )
    dD = dC;

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    printf("\nRunning with cuSPARSE...\n");
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;
    CHECK_CUSPARSE( cusparseLtInit(&handle) )
    // matrix descriptor initialization
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &handle, &matA, num_A_rows,
                                            num_A_cols, lda, alignment,
                                            A_device_dtype, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matB, num_B_rows,
                                            num_B_cols, ldb, alignment,
                                            B_device_dtype, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            C_device_dtype, order) )
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &handle, &matmul, opA, opB,
                                            &matA, &matB, &matC, &matC,
                                            compute_type_sparse) )
    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )
    int alg = 0;
    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(
                                            &handle, &alg_sel,
                                            CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                            &alg, sizeof(alg)))
    size_t workspace_size;
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                             workspace_size) )
    //--------------------------------------------------------------------------
    // Prune the A matrix (in-place) and check the correctness
    elapse_time = 0;
    cudaErrCheck( cudaEventRecord(start));
    CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
                                         CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
                                              d_valid, stream) )
    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&elapse_time, start, stop));
    printf("matrix pruning took %fms\n", elapse_time);
    int is_valid;
    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
                                cudaMemcpyDeviceToHost, stream) )
    CHECK_CUDA( cudaStreamSynchronize(stream) )
    if (is_valid != 0) {
        std::printf("!!!! The matrix has been pruned in a wrong way. "
                    "cusparseLtMatmul will not provide correct results\n");
        return EXIT_FAILURE;
    }
    //--------------------------------------------------------------------------
    // Compress the A matrix
    size_t compressed_size;

    elapse_time = 0;
    cudaErrCheck( cudaEventRecord(start));
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
                                                  &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA,
                                            dA_compressed, stream) )
    cudaErrCheck( cudaEventRecord(stop));
    cudaErrCheck( cudaEventSynchronize(stop));
    cudaErrCheck( cudaEventElapsedTime(&elapse_time, start, stop));
    printf("matrix compression took %fms\n", elapse_time);
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;

    elapse_time = 0;
    cudaErrCheck( cudaEventRecord(start));
    CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha,
                                           dA_compressed, dB, &beta,
                                           dC, dD, d_workspace,
                                           streams, num_streams) )
    cudaErrCheck( cudaEventRecord(stop));
    cudaErrCheck( cudaEventSynchronize(stop));
    cudaErrCheck( cudaEventElapsedTime(&elapse_time, start, stop));
    printf("matmual search took %fms\n", elapse_time);

    int alg_id;
    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                           &handle, &alg_sel,
                                           CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                           &alg_id, sizeof(alg_id)) )
    int32_t splitK, splitKBuffers;
    cusparseLtSplitKMode_t splitKMode;

    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                           &handle, &alg_sel,
                                           CUSPARSELT_MATMUL_SPLIT_K,
                                           &splitK, sizeof(splitK)) )

    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                           &handle, &alg_sel,
                                           CUSPARSELT_MATMUL_SPLIT_K_MODE,
                                           &splitKMode, sizeof(splitKMode)) )

    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                           &handle, &alg_sel,
                                           CUSPARSELT_MATMUL_SPLIT_K_BUFFERS,
                                           &splitKBuffers,
                                           sizeof(splitKBuffers)) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                             workspace_size) )

    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &plan,
                                                 &workspace_size))

    CHECK_CUDA( cudaMalloc((void**)&d_workspace, workspace_size) )
    // Perform the matrix multiplication
    float average_time = 0;
    for (int i = 0; i < n_iter; i++) {
        elapse_time = 0;
        cudaErrCheck( cudaMemset(dC, 0, C_size)); // clear c buffer to zero
        cudaErrCheck( cudaEventRecord(start));
        CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                        &beta, dC, dD, d_workspace, streams,
                                        num_streams) )
        cudaErrCheck( cudaEventRecord(stop));
        cudaErrCheck( cudaEventSynchronize(stop));
        cudaErrCheck( cudaEventElapsedTime(&elapse_time, start, stop));
        average_time += elapse_time;
    }
    average_time = average_time / n_iter;
    printf("sparse matrix multiplication average "
        "took %fms\n over %d iterations\n", average_time, n_iter);
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // destroy plan and handle
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )
    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )
    //--------------------------------------------------------------------------
    // device result check

    if (check_result) {
        // matrix A has been pruned, so we copied pruned A from device to host
        // !!!!! This will modify the original value in matrix A !!!!!!
        CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )
        CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )

        bool A_std_layout = (is_rowmajor != isA_transposed);
        bool B_std_layout = (is_rowmajor != isB_transposed);

        // host computation
        float* hC_result = (float*) malloc(sizeof(float)*m*n);
        memset(hC_result, 0, sizeof(float)*m*n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum  = 0.0f;
                for (int k1 = 0; k1 < k; k1++) {
                    auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
                    auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
                    sum      += static_cast<float>(hA[posA]) *  // [i][k]
                                static_cast<float>(hB[posB]);   // [k][j]
                }
                auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                hC_result[posC] = sum;  // [i][j]
            }
        }

        // print if verbose > 2
        if (verbose > 2) {
            printf("hA\n");
            for (int i = 0; i < m; i++) {
                for (int k1 = 0; k1 < k; k1++) {
                    auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
                    printf("%f ", static_cast<float>(hA[posA]));
                }
                printf("\n");
            }
            printf("hB\n");
            for (int k1 = 0; k1 < k; k1++) {
                for (int j = 0; j < n; j++) {
                    auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
                    printf("%f ", static_cast<float>(hB[posB]));
                }
                printf("\n");
            }
            printf("hC_result\n");
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    auto pos = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                    printf("%f ", static_cast<float>(hC_result[pos]));
                }
                printf("\n");
            }
            printf("hC\n");
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    auto pos = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                    printf("%f ", static_cast<float>(hC[pos]));
                }
                printf("\n");
            }
        }

        // host-device comparison
        int correct = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                auto device_value = static_cast<float>(hC[pos]);
                auto host_value   = hC_result[pos];
                // direct floating point comparison is not reliable
                // if (device_value != host_value) {
                if (fabs(device_value - host_value) > 0.1 && verbose > 1) {
                    std::printf("(%d, %d):\t%f vs. %f\n",
                                i, j, host_value, device_value);
                    correct = 0;
                    break;
                }
            }
        }
        if (correct)
            std::printf("sparse_matmul test PASSED\n");
        else
            std::printf("sparse_matmul test FAILED: wrong result\n");
        // free memory    
        free(hC_result);
    }
    
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )
    CHECK_CUDA( cudaFree(d_workspace) )

    return average_time;
}

float profile_dense_tensor_core_matmul(
    int n_iter, A_DTYPE * hA, B_DTYPE *hB, C_DTYPE *hC, float alpha, float beta, bool check_result, int verbose = 0) {
    float elapse_time = 0;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaErrCheck( cudaEventCreate(&start));
    cudaErrCheck( cudaEventCreate(&stop));
    //--------------------------------------------------------------------------
    // Device memory management
    A_DTYPE *dA;
    B_DTYPE *dB;
    C_DTYPE *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    printf("\nRunning with cuBLAS...\n");
    cublasHandle_t cublasHandle;
    cublasErrCheck( cublasCreate(&cublasHandle));
    cublasErrCheck(
        cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH)
    ); // Use tensor cores
    printf("Tensor cores ENABLED\n");
    // Now using cuBLAS
    // Perform the matrix multiplication
    float average_time = 0;
    for (int i = 0; i < n_iter; i++) {
        elapse_time = 0;
        cudaErrCheck( cudaMemset(dC, 0, C_size)); // clear c buffer to zero
        cudaErrCheck( cudaEventRecord(start));
        // A transposed and B non-transposed to use tensor core
        cublasErrCheck( cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
        // cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        m, n, k, 
                        &alpha,
                        dA, A_device_dtype, m,
                        dB, B_device_dtype, k,
                        &beta, 
                        dC, C_device_dtype, m,
                        compute_type_cublas_tc, CUBLAS_GEMM_DFALT_TENSOR_OP));
        cudaErrCheck( cudaEventRecord(stop));
        cudaErrCheck( cudaEventSynchronize(stop));
        cudaErrCheck( cudaEventElapsedTime(&elapse_time, start, stop));
        average_time += elapse_time;
    }
    average_time = average_time / n_iter;
    printf("dense matrix multiplication with tensor core average "
        "took %fms over %d iterations\n", average_time, n_iter);

    if (check_result) {
        // cuBLAS and most other linear algebra libraries use the column-major format
        // https://forums.developer.nvidia.com/t/cublassgemm-v2-returns-incorrect-matrix-multiplication-results/28601
        // but we config cublasGemmEx to CUBLAS_OP_T, CUBLAS_OP_N
        CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )
        bool is_rowmajor = false;       // linear algebra libraries use the column-major
        bool isA_transposed = true;     // CUBLAS_OP_T for A
        bool isB_transposed = false;    // CUBLAS_OP_N for B
        bool A_std_layout = (is_rowmajor != isA_transposed);
        bool B_std_layout = (is_rowmajor != isB_transposed);
        // // host computation
        float* hC_result = (float*) malloc(sizeof(float)*m*n);
        memset(hC_result, 0, sizeof(float)*m*n);


        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum  = 0.0f;
                for (int k1 = 0; k1 < k; k1++) {
                    auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
                    auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
                    sum      += static_cast<float>(hA[posA]) *  // [i][k]
                                static_cast<float>(hB[posB]);   // [k][j]
                }
                auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                hC_result[posC] = sum;  // [i][j]
            }
        }


        // print if verbose > 2
        if (verbose > 2) {
            printf("hA\n");
            for (int i = 0; i < m; i++) {
                for (int k1 = 0; k1 < k; k1++) {
                    auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
                    printf("%f ", static_cast<float>(hA[posA]));
                }
                printf("\n");
            }
            printf("hB\n");
            for (int k1 = 0; k1 < k; k1++) {
                for (int j = 0; j < n; j++) {
                    auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
                    printf("%f ", static_cast<float>(hB[posB]));
                }
                printf("\n");
            }
            printf("hC_result\n");
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    auto pos = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                    printf("%f ", static_cast<float>(hC_result[pos]));
                }
                printf("\n");
            }
            printf("hC\n");
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    auto pos = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                    printf("%f ", static_cast<float>(hC[pos]));
                }
                printf("\n");
            }
        }
       
        // host-device comparison
        int correct = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                // auto pos          = i + j * ldc;
                auto device_value = static_cast<float>(hC[pos]);
                auto host_value   = hC_result[pos];
                // direct floating point comparison is not reliable
                // if (device_value != host_value) {
                if (fabs(device_value - host_value) > 0.1 && verbose > 1) {
                    std::printf("(%d, %d):\t%f vs. %f\n",
                                i, j, host_value, device_value);
                    correct = 0;
                    break;
                }
            }
        }
        if (correct)
            std::printf("dense_matmul test PASSED\n");
        else
            std::printf("dense_matmul test FAILED: wrong result\n");
        // free memory    
        free(hC_result);
    }

    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )

    return average_time;
}

float profile_dense_matmul(
    int n_iter, A_DTYPE * hA, B_DTYPE *hB, C_DTYPE *hC, float alpha, float beta, bool check_result, int verbose = 0) {
    float elapse_time = 0;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaErrCheck( cudaEventCreate(&start));
    cudaErrCheck( cudaEventCreate(&stop));
    //--------------------------------------------------------------------------
    // Device memory management
    A_DTYPE *dA;
    B_DTYPE *dB;
    C_DTYPE *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------

    printf("\nRunning with cuBLAS...\n");
    cublasHandle_t cublasHandle;
    cublasErrCheck( cublasCreate(&cublasHandle));
    cublasErrCheck(
        cublasSetMathMode(cublasHandle, CUBLAS_PEDANTIC_MATH)
    ); // Disable tensor cores
    // Now using cuBLAS
    printf("Tensor cores DISABLED\n");
    // Perform the matrix multiplication
    float average_time = 0;
    for (int i = 0; i < n_iter; i++) {
        elapse_time = 0;
        cudaErrCheck( cudaMemset(dC, 0, C_size)); // clear c buffer to zero
        cudaErrCheck( cudaEventRecord(start));
        // A transposed and B non-transposed to use tensor core
        // but it is useless here
        cublasErrCheck( cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
        // cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        m, n, k, 
                        &alpha,
                        dA, A_device_dtype, m,
                        dB, B_device_dtype, k,
                        &beta, 
                        dC, C_device_dtype, m,
                        compute_type_cublas, CUBLAS_GEMM_DEFAULT));
        cudaErrCheck(cudaEventRecord(stop));
        cudaErrCheck(cudaEventSynchronize(stop));
        cudaErrCheck(cudaEventElapsedTime(&elapse_time, start, stop));
        average_time += elapse_time;
    }
    average_time = average_time / n_iter;
    printf("dense matrix multiplication average took "
        "%fms over %d iterations\n", average_time, n_iter);

    if (check_result) {
        // cuBLAS and most other linear algebra libraries use the column-major format
        // https://forums.developer.nvidia.com/t/cublassgemm-v2-returns-incorrect-matrix-multiplication-results/28601
        // but we config cublasGemmEx to CUBLAS_OP_T, CUBLAS_OP_N
        CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )
        bool is_rowmajor = false;       // linear algebra libraries use the column-major
        bool isA_transposed = true;     // CUBLAS_OP_T for A
        bool isB_transposed = false;    // CUBLAS_OP_N for B
        bool A_std_layout = (is_rowmajor != isA_transposed);
        bool B_std_layout = (is_rowmajor != isB_transposed);
        // // host computation
        float* hC_result = (float*) malloc(sizeof(float)*m*n);
        memset(hC_result, 0, sizeof(float)*m*n);


        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum  = 0.0f;
                for (int k1 = 0; k1 < k; k1++) {
                    auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
                    auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
                    sum      += static_cast<float>(hA[posA]) *  // [i][k]
                                static_cast<float>(hB[posB]);   // [k][j]
                }
                auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                hC_result[posC] = sum;  // [i][j]
            }
        }


        // print if verbose > 2
        if (verbose > 2) {
            printf("hA\n");
            for (int i = 0; i < m; i++) {
                for (int k1 = 0; k1 < k; k1++) {
                    auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
                    printf("%f ", static_cast<float>(hA[posA]));
                }
                printf("\n");
            }
            printf("hB\n");
            for (int k1 = 0; k1 < k; k1++) {
                for (int j = 0; j < n; j++) {
                    auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
                    printf("%f ", static_cast<float>(hB[posB]));
                }
                printf("\n");
            }
            printf("hC_result\n");
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    auto pos = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                    printf("%f ", static_cast<float>(hC_result[pos]));
                }
                printf("\n");
            }
            printf("hC\n");
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    auto pos = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                    printf("%f ", static_cast<float>(hC[pos]));
                }
                printf("\n");
            }
        }
       
        // host-device comparison
        int correct = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
                // auto pos          = i + j * ldc;
                auto device_value = static_cast<float>(hC[pos]);
                auto host_value   = hC_result[pos];
                // direct floating point comparison is not reliable
                // if (device_value != host_value) {
                if (fabs(device_value - host_value) > 0.1 && verbose > 1) {
                    std::printf("(%d, %d):\t%f vs. %f\n",
                                i, j, host_value, device_value);
                    correct = 0;
                    break;
                }
            }
        }
        if (correct)
            std::printf("dense_matmul test PASSED\n");
        else
            std::printf("dense_matmul test FAILED: wrong result\n");
        // free memory    
        free(hC_result);
    }
    
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )

    return average_time;
}


int main(void) {
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                       cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                       cudaDevAttrComputeCapabilityMinor, 0) )
    if (!(major_cc == 8 && minor_cc == 0) &&
        !(major_cc == 8 && minor_cc == 6)) {
        std::printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability == 8.0, 8.6 current: %d.%d\n\n",
                     major_cc, minor_cc);
        return EXIT_UNSUPPORTED;
    }

    printf("Matmul size: %dx%dx%d\n", m, k, n);
    printf("Create matrices with sparse ratio: %f\n", sparse_ratio);
    printf("Number of iterations: %d\n", n_iter);

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> unifrom(0, 1);  // uniform distribution in [0, 1]
    std::normal_distribution<> normal{0, 1};  // normal distribution in mean 0, std 1

    // allocate memory in host DRAM
    A_DTYPE *hA;
    B_DTYPE *hB;
    C_DTYPE *hC;
    hA = (A_DTYPE*) malloc(sizeof(A_DTYPE)*m*k);
    hB = (B_DTYPE*) malloc(sizeof(B_DTYPE)*k*n);
    hC = (C_DTYPE*) malloc(sizeof(C_DTYPE)*m*n);  // place gpu result
    memset(hC, 0, sizeof(C_DTYPE)*m*n);

    for (int i = 0; i < m * k; i++) {
        if (unifrom(gen) > sparse_ratio) {
            #if defined(I8)
                hA[i] = static_cast<A_DTYPE>(normal(gen) * 255 - 127);
            #else
                hA[i] = static_cast<A_DTYPE>(normal(gen));
            #endif
        } else {
            hA[i] = 0.0;
        }
    }
    for (int i = 0; i < k * n; i++) {
        if (unifrom(gen) > sparse_ratio) {
            #if defined(I8)
                hB[i] = static_cast<B_DTYPE>(normal(gen) * 255 - 127);
            #else
                hB[i] = static_cast<B_DTYPE>(normal(gen));
            #endif
        } else {
            hB[i] = 0.0;
        }
    }
    float alpha = 1.0f;
    float beta  = 0.0f;


    // !!!!! Matrix A will be pruned in profile_sparse_matmul !!!!!
    memset(hC, 0, sizeof(C_DTYPE)*m*n);
    float sparse_time =  profile_sparse_matmul(n_iter, hA, hB, hC, alpha, beta, true, 0);

    memset(hC, 0, sizeof(C_DTYPE)*m*n);
    float dense_tc_time =  profile_dense_tensor_core_matmul(n_iter, hA, hB, hC, alpha, beta, true, 0);
    
    memset(hC, 0, sizeof(C_DTYPE)*m*n);
    float dense_time =  profile_dense_matmul(n_iter, hA, hB, hC, alpha, beta, true, 0);

    free(hA);
    free(hB);
    free(hC);
    return EXIT_SUCCESS;
}
