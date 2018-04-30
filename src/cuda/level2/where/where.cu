#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include <stdio.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
    
float kernelTime = 0.0f;
float transferTime = 0.0f;
cudaEvent_t start, stop;
float elapsedTime;

__device__ bool check(int val, int bound) {
    return (val < bound);
}

__global__ void markMatches(int *arr, int *results, int size, int bound) {

    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;

    int tid = (blockDim.x * bx) + tx;

    for( ; tid < size; tid += blockDim.x * gridDim.x) {
        if(check(arr[tid], bound)) {
            results[tid] = 1;
        } else {
            results[tid] = 0;
        }
    }
}

__global__ void mapMatches(int *arr, int *results, int *prefix, int *final, int size) {

    // Block index
    int bx = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;

    int tid = (blockDim.x * bx) + tx;

    for( ; tid < size; tid += blockDim.x * gridDim.x) {
        if(results[tid]) {
            final[prefix[tid]] = arr[tid];
        }
    }
}

void seedArr(int *arr, int size) {
    for(int i = 0; i < size; i++) {
        arr[i] = rand() % 100;
    }
}

void where(ResultDatabase &resultDB, int size, int coverage) {

    int *arr = (int*)malloc(sizeof(int) * size);
    int *final;
    seedArr(arr, size);

    int *d_arr;
    int *d_results;
    int *d_prefix;
    int *d_final;
    
    CUDA_SAFE_CALL(cudaMalloc( (void**) &d_arr, sizeof(int) * size));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &d_results, sizeof(int) * size));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &d_prefix, sizeof(int) * size));

    cudaEventRecord(start, 0);
    cudaMemcpy(d_arr, arr, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    transferTime += elapsedTime * 1.e-3;

    dim3 grid(size / 1024 + 1, 1, 1);
    dim3 threads(1024, 1, 1);
    cudaEventRecord(start, 0);
    markMatches<<<grid, threads>>>(d_arr, d_results, size, coverage);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();

    cudaEventRecord(start, 0);
    thrust::exclusive_scan(thrust::device, d_results, d_results + size, d_prefix);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();

    int matchSize;
    cudaEventRecord(start, 0);
    cudaMemcpy(&matchSize, d_prefix + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    transferTime += elapsedTime * 1.e-3;
    matchSize++;

    CUDA_SAFE_CALL(cudaMalloc( (void**) &d_final, sizeof(int) * matchSize));
    final = (int*)malloc(sizeof(int) * matchSize);

    cudaEventRecord(start, 0);
    mapMatches<<<grid, threads>>>(d_arr, d_results, d_prefix, d_final, size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();

    cudaEventRecord(start, 0);
    cudaMemcpy(final, d_final, sizeof(int) * matchSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    transferTime += elapsedTime * 1.e-3;

    free(arr);
    free(final);
    CUDA_SAFE_CALL(cudaFree(d_arr));
    CUDA_SAFE_CALL(cudaFree(d_results));
    CUDA_SAFE_CALL(cudaFree(d_prefix));
    CUDA_SAFE_CALL(cudaFree(d_final));
    
    char atts[1024];
    sprintf(atts, "size:%d, coverage:%d", size, coverage);
    resultDB.AddResult("where_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("where_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("where_parity", atts, "N", transferTime / kernelTime);
}

void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("length", OPT_INT, "0", "number of elements in input");
  op.addOption("coverage", OPT_INT, "-1", "0 to 100 percentage of elements to allow through where filter");
}

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    srand(7);
    int size = op.getOptionInt("length");
    int coverage = op.getOptionInt("coverage");
    if(size == 0 || coverage == -1) {
        int sizes[4] = {1000, 10000, 500000000, 1000000000};
        int coverages[4] = {20, 30, 40, 80};
        size = sizes[op.getOptionInt("size") - 1];
        coverage = coverages[op.getOptionInt("size") - 1];
    }

    printf("Using size=%d, coverage=%d\n", size, coverage);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int passes = op.getOptionInt("passes");
    for(int i = 0; i < passes; i++) {
        kernelTime = 0.0f;
        transferTime = 0.0f;
        printf("Pass %d: ", i);
        where(resultDB, size, coverage);
        printf("Done.\n");
    }

}
