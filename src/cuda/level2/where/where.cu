////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\where\where.cu
//
// summary:	Where class
// 
// origin: 
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include <stdio.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
    
/// <summary>	The kernel time. </summary>
float kernelTime = 0.0f;
/// <summary>	The transfer time. </summary>
float transferTime = 0.0f;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the stop. </summary>
///
/// <value>	The stop. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

cudaEvent_t start, stop;
/// <summary>	The elapsed time. </summary>
float elapsedTime;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Checks. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="val">  	The value. </param>
/// <param name="bound">	The bound. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ bool check(int val, int bound) {
    return (val < bound);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Mark matches. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="arr">	  	[in,out] If non-null, the array. </param>
/// <param name="results">	[in,out] If non-null, the results. </param>
/// <param name="size">   	The size. </param>
/// <param name="bound">  	The bound. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Map matches. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="arr">	  	[in,out] If non-null, the array. </param>
/// <param name="results">	[in,out] If non-null, the results. </param>
/// <param name="prefix"> 	[in,out] If non-null, the prefix. </param>
/// <param name="final">  	[in,out] If non-null, the final. </param>
/// <param name="size">   	The size. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Seed array. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="arr"> 	[in,out] If non-null, the array. </param>
/// <param name="size">	The size. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void seedArr(int *arr, int size) {
    for(int i = 0; i < size; i++) {
        arr[i] = rand() % 100;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Wheres. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="size">	   	The size. </param>
/// <param name="coverage">	The coverage. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void where(ResultDatabase &resultDB, OptionParser &op, int size, int coverage) {
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    int device = 0;
    checkCudaErrors(cudaGetDevice(&device));

    int *arr = NULL;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(cudaMallocManaged(&arr, sizeof(int) * size));
    } else {
        arr = (int*)malloc(sizeof(int) * size);
        assert(arr);
    }
    int *final;
    seedArr(arr, size);

    int *d_arr;
    int *d_results;
    int *d_prefix;
    int *d_final;
    
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        d_arr = arr;
        checkCudaErrors(cudaMallocManaged( (void**) &d_results, sizeof(int) * size));
        checkCudaErrors(cudaMallocManaged( (void**) &d_prefix, sizeof(int) * size));
    } else {
        checkCudaErrors(cudaMalloc( (void**) &d_arr, sizeof(int) * size));
        checkCudaErrors(cudaMalloc( (void**) &d_results, sizeof(int) * size));
        checkCudaErrors(cudaMalloc( (void**) &d_prefix, sizeof(int) * size));
    }

    checkCudaErrors(cudaEventRecord(start, 0));
    if (uvm) {
        // do nothing
    } else if (uvm_advise) {
        checkCudaErrors(cudaMemAdvise(d_arr, sizeof(int) * size, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(d_arr, sizeof(int) * size, cudaMemAdviseSetPreferredLocation, device));
    } else if (uvm_prefetch) {
        checkCudaErrors(cudaMemPrefetchAsync(d_arr, sizeof(int) * size, device));
    } else if (uvm_prefetch_advise) {
        checkCudaErrors(cudaMemAdvise(d_arr, sizeof(int) * size, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(d_arr, sizeof(int) * size, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemPrefetchAsync(d_arr, sizeof(int) * size, device));
    } else {
        checkCudaErrors(cudaMemcpy(d_arr, arr, sizeof(int) * size, cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    transferTime += elapsedTime * 1.e-3;

    dim3 grid(size / 1024 + 1, 1, 1);
    dim3 threads(1024, 1, 1);
    checkCudaErrors(cudaEventRecord(start, 0));
    markMatches<<<grid, threads>>>(d_arr, d_results, size, coverage);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();

    checkCudaErrors(cudaEventRecord(start, 0));
    thrust::exclusive_scan(thrust::device, d_results, d_results + size, d_prefix);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();

    int matchSize;
    checkCudaErrors(cudaEventRecord(start, 0));
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        matchSize = (int)*(d_prefix + size - 1);
    } else {
        checkCudaErrors(cudaMemcpy(&matchSize, d_prefix + size - 1, sizeof(int), cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    transferTime += elapsedTime * 1.e-3;
    matchSize++;

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(cudaMallocManaged( (void**) &d_final, sizeof(int) * matchSize));
        final = d_final;
    } else {
        checkCudaErrors(cudaMalloc( (void**) &d_final, sizeof(int) * matchSize));
        final = (int*)malloc(sizeof(int) * matchSize);
        assert(final);
    }

    checkCudaErrors(cudaEventRecord(start, 0));
    mapMatches<<<grid, threads>>>(d_arr, d_results, d_prefix, d_final, size);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();

    checkCudaErrors(cudaEventRecord(start, 0));
    // No cpy just demand paging
    if (uvm) {
        // Do nothing
    } else if (uvm_advise) {
        checkCudaErrors(cudaMemAdvise(final, sizeof(int) * matchSize, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(final, sizeof(int) * matchSize, cudaMemAdviseSetPreferredLocation, device));
    } else if (uvm_prefetch) {
        checkCudaErrors(cudaMemPrefetchAsync(final, sizeof(int) * matchSize, cudaCpuDeviceId));
    } else if (uvm_prefetch_advise) {
        checkCudaErrors(cudaMemAdvise(final, sizeof(int) * matchSize, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(final, sizeof(int) * matchSize, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemPrefetchAsync(final, sizeof(int) * matchSize, cudaCpuDeviceId));
    } else {
        checkCudaErrors(cudaMemcpy(final, d_final, sizeof(int) * matchSize, cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    transferTime += elapsedTime * 1.e-3;

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(cudaFree(d_arr));
        checkCudaErrors(cudaFree(d_results));
        checkCudaErrors(cudaFree(d_prefix));
        checkCudaErrors(cudaFree(d_final));
    } else {
        free(arr);
        free(final);
        checkCudaErrors(cudaFree(d_arr));
        checkCudaErrors(cudaFree(d_results));
        checkCudaErrors(cudaFree(d_prefix));
        checkCudaErrors(cudaFree(d_final));
    }
    
    char atts[1024];
    sprintf(atts, "size:%d, coverage:%d", size, coverage);
    resultDB.AddResult("where_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("where_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("where_total_time", atts, "sec", kernelTime+transferTime);
    resultDB.AddResult("where_parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime+transferTime);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds a benchmark specifier options. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="op">	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("length", OPT_INT, "0", "number of elements in input");
  op.addOption("coverage", OPT_INT, "-1", "0 to 100 percentage of elements to allow through where filter");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    printf("Running Where\n");

    srand(7);

    bool quiet = op.getOptionBool("quiet");
    int size = op.getOptionInt("length");
    int coverage = op.getOptionInt("coverage");
    if (size == 0 || coverage == -1) {
        int sizes[5] = {1000, 10000, 500000000, 1000000000, 1050000000};
        int coverages[5] = {20, 30, 40, 80, 240};
        size = sizes[op.getOptionInt("size") - 1];
        coverage = coverages[op.getOptionInt("size") - 1];
    }

    if (!quiet) {
        printf("Using size=%d, coverage=%d\n", size, coverage);
    }

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    int passes = op.getOptionInt("passes");
    for (int i = 0; i < passes; i++) {
        kernelTime = 0.0f;
        transferTime = 0.0f;
        if(!quiet) {
            printf("Pass %d: ", i);
        }
        where(resultDB, op, size, coverage);
        if(!quiet) {
            printf("Done.\n");
        }
    }
}
