////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level0\devicememory\DeviceMemory.cu
//
// summary:	Device memory class
// 
// origin: SHOC (https://github.com/vetter/shoc)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OptionParser.h"
#include "ResultDatabase.h"
// #include "Timer.h"
#include "Utility.h"
#include "cudacommon.h"
#include "support.h"
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	 Forward declarations for texture memory test and benchmark kernels.
/// 			 Tests texture memory. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation specified by the user in command line. </param>
/// <param name="scalet">  	The scalet. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void TestTextureMem(ResultDatabase &resultDB, OptionParser &op, double scalet);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads global memory coalesced. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size of input data. </param>
/// <param name="repeat">	The repeat times. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void readGlobalMemoryCoalesced(float *data, float *output, int globalWorkSize, int size, int repeat);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads global memory unit by unit. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size of input data. </param>
/// <param name="repeat">	The repeat times. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void readGlobalMemoryUnit(float *data, float *output, int maxGroupSize, int size, int repeat);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads local memory. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="data">  	The data. </param>
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size pf input data. </param>
/// <param name="repeat">	The repeat times. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void readLocalMemory(const float *data, float *output, int size, int repeat);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes to global memory coalesced. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size of input data. </param>
/// <param name="repeat">	The repeat times for specified operations. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void writeGlobalMemoryCoalesced(float *output, int globalWorkSize, int size, int repeat);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes to global memory unit by unit. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size of data. </param>
/// <param name="repeat">	The repeat times of global write. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void writeGlobalMemoryUnit(float *output, int maxGroupSize, int size, int repeat);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes to local memory. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size. </param>
/// <param name="repeat">	The repeat times of write operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void writeLocalMemory(float *output, int size, int repeat);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets a random int (GPU). </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="seed">	The seed. </param>
/// <param name="mod"> 	The modifier. </param>
///
/// <returns>	The randomly generated int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int getRand(int seed, int mod);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads the texels. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="n">		An int to process. </param>
/// <param name="d_out">	[in,out] If non-null, the out. </param>
/// <param name="width">	The width. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void readTexels(int n, float *d_out, int width);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads texels in cache. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="n">		An int to process. </param>
/// <param name="d_out">	[in,out] If non-null, the out. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void readTexelsInCache(int n, float *d_out);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads texels random. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="n">	 	An int to process. </param>
/// <param name="d_out"> 	[in,out] If non-null, the out. </param>
/// <param name="width"> 	The width. </param>
/// <param name="height">	The height. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void readTexelsRandom(int n, float *d_out, int width, int height);
// Texture to use for the benchmarks
/// <summary>	The tex a. </summary>
texture<float4, 2, cudaReadModeElementType> texA;

// ****************************************************************************
// Function: addBenchmarkSpecOptions (From SHOC)
//
// Purpose:
//   Add benchmark specific options parsing.  Note that device memory has no
//   benchmark specific options, so this is just a stub.
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 11, 2009
//
// Modifications:
//
// ****************************************************************************

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("uvm", OPT_BOOL, "0", "enable CUDA Unified Virtual Memory, only use demand paging");
}

// ****************************************************************************
// Function: runBenchmark (From SHOC)
//
// Purpose:
//   This benchmark measures the device memory bandwidth for several areas
//   of memory including global, shared, and texture memories for several
//   types of access patterns.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: September 08, 2009
//
// Modifications:
//   Gabriel Marin, 06/09/2010: Change memory access patterns to eliminate
//   data reuse. Add auto-scaling factor.
//
//   Jeremy Meredith, 10/09/2012: Ignore errors at large thread counts
//   in case only smaller thread counts succeed on some devices.
//
//   Jeremy Meredith, Wed Oct 10 11:54:32 EDT 2012
//   Auto-scaling factor could be less than 1 on some problems.  This would
//   make some iteration counts zero and actually skip tests.  I enforced
//   that the factor ber at least 1.
//   
//   Bodun Hu, May 29 2020: added safe call wrappers
//   Update deprecated calls
//   Update threads and blocks dim
//
// ****************************************************************************

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    cout << "Running DeviceMemory" << endl;
    // Enable quiet output
    bool quiet = op.getOptionBool("quiet");
    // Number of times to repeat each test
    const unsigned int passes = op.getOptionInt("passes");
    const bool uvm = op.getOptionBool("uvm");

    // threads dim per block
    size_t minGroupSize = warp_size(op.getOptionInt("device"));
    size_t maxGroupSize = max_threads_per_block(op.getOptionInt("device"));

    size_t globalWorkSize = 64 * maxGroupSize;  // 64 * 1024=65536, 64 blocks, may need tuning
    unsigned long memSize = 1024 * 1024 * 1024; // 64MB buffer
    // unsigned long memSize = ((unsigned long)-1) / 2;
    void *testmem = NULL;
    cudaMalloc(&testmem, memSize * 2);
    while (cudaGetLastError() != cudaSuccess && memSize != 0) {
        memSize >>= 1; // keept it a power of 2
        cudaMalloc(&testmem, memSize * 2);
    }
    checkCudaErrors(cudaFree(testmem));
    if (memSize == 0) {
        printf("Not able to allocate device memory. Exiting!\n");
        safe_exit(-1);
    }

    const unsigned int numWordsFloat = memSize / sizeof(float);

    // Initialize host memory
    float *h_in = new float[numWordsFloat];
    float *h_out = new float[numWordsFloat];
    srand48(8650341L);
    for (int i = 0; i < numWordsFloat; ++i) {
        h_in[i] = (float)(drand48() * numWordsFloat);
    }

    // Allocate two chunks of device memory
    float *d_mem1, *d_mem2;
    char sizeStr[128];

    if (uvm) {
        checkCudaErrors(cudaMallocManaged((void **)&d_mem1, sizeof(float) * (numWordsFloat)));
        checkCudaErrors(cudaMallocManaged((void **)&d_mem2, sizeof(float) * (numWordsFloat)));
    } else {
        checkCudaErrors(cudaMalloc((void **)&d_mem1, sizeof(float) * (numWordsFloat)));
        checkCudaErrors(cudaMalloc((void **)&d_mem2, sizeof(float) * (numWordsFloat)));
    }

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, 0));
    readGlobalMemoryCoalesced<<<512, 64>>>(d_mem1, d_mem2, 512*64, numWordsFloat, 256);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float t = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&t, start, stop));
    t /= 1.e3;
    double scalet = 0.1 / t;
    if (scalet < 1)
        scalet = 1;

    const unsigned int maxRepeatsCoal = 256 * scalet;
    const unsigned int maxRepeatsUnit = 16 * scalet;
    const unsigned int maxRepeatsLocal = 300 * scalet;

    for (int p = 0; p < passes; p++) {
        // Run the kernel for each group size
        if (!quiet) {
            cout << "Pass: " << p << "\n";
        }
        for (int threads = minGroupSize; threads <= maxGroupSize; threads *= 2) {
            const unsigned int blocks = globalWorkSize / threads;
            assert(globalWorkSize % threads == 0);     // Just to make sure no outstanding thread
            double bdwth;
            sprintf(sizeStr, "blockSize:%03d", threads);

            // Test 1
            checkCudaErrors(cudaEventRecord(start, 0));

            readGlobalMemoryCoalesced<<<blocks, threads>>>(
                    d_mem1, d_mem2, globalWorkSize, numWordsFloat, maxRepeatsCoal);
                    
            checkCudaErrors(cudaEventRecord(stop, 0));
            checkCudaErrors(cudaEventSynchronize(stop));

            // We can run out of resources at larger thread counts on
            // some devices.  If we made a successful run at smaller
            // thread counts, just ignore errors at this size.
            if (threads > minGroupSize) {
                if (cudaGetLastError() != cudaSuccess)
                    break;
            } else {
                CHECK_CUDA_ERROR();
            }
            t = 0.0f;
            checkCudaErrors(cudaEventElapsedTime(&t, start, stop));
            t /= 1.e3;
            bdwth = ((double)globalWorkSize * maxRepeatsCoal * 16 * sizeof(float)) /
                            (t * 1000. * 1000. * 1000.);
            resultDB.AddResult("readGlobalMemoryCoalesced", sizeStr, "GB/s", bdwth);

            // Test 2
            checkCudaErrors(cudaEventRecord(start, 0));

            readGlobalMemoryUnit<<<blocks, threads>>>(d_mem1, d_mem2, maxGroupSize, numWordsFloat, maxRepeatsUnit);

            checkCudaErrors(cudaEventRecord(stop, 0));
            checkCudaErrors(cudaEventSynchronize(stop));
            CHECK_CUDA_ERROR();
            checkCudaErrors(cudaEventElapsedTime(&t, start, stop));
            t /= 1.e3;
            bdwth = ((double)globalWorkSize * maxRepeatsUnit * 16 * sizeof(float)) /
                            (t * 1000. * 1000. * 1000.);
            resultDB.AddResult("readGlobalMemoryUnit", sizeStr, "GB/s", bdwth);
            resultDB.AddOverall("Bandwidth", "GB/s", bdwth);

            // Test 3
            checkCudaErrors(cudaEventRecord(start, 0));

            readLocalMemory<<<blocks, threads>>>(d_mem1, d_mem2, numWordsFloat, maxRepeatsLocal);

            checkCudaErrors(cudaEventRecord(stop, 0));
            checkCudaErrors(cudaEventSynchronize(stop));
            CHECK_CUDA_ERROR();
            checkCudaErrors(cudaEventElapsedTime(&t, start, stop));
            t /= 1.e3;
            bdwth = ((double)globalWorkSize * maxRepeatsLocal * 16 * sizeof(float)) /
                            (t * 1000. * 1000. * 1000.);
            resultDB.AddResult("readLocalMemory", sizeStr, "GB/s", bdwth);

            // Test 4
            checkCudaErrors(cudaEventRecord(start, 0));

            writeGlobalMemoryCoalesced<<<blocks, threads>>>(d_mem2, globalWorkSize, numWordsFloat, maxRepeatsCoal);

            checkCudaErrors(cudaEventRecord(stop, 0));
            checkCudaErrors(cudaEventSynchronize(stop));
            CHECK_CUDA_ERROR();
            checkCudaErrors(cudaEventElapsedTime(&t, start, stop));
            t /= 1.e3;
            bdwth = ((double)globalWorkSize * maxRepeatsCoal * 16 * sizeof(float)) /
                            (t * 1000. * 1000. * 1000.);
            resultDB.AddResult("writeGlobalMemoryCoalesced", sizeStr, "GB/s", bdwth);

            // Test 5
            checkCudaErrors(cudaEventRecord(start, 0));

            writeGlobalMemoryUnit<<<blocks, threads>>>(d_mem2, maxGroupSize, numWordsFloat, maxRepeatsUnit);

            checkCudaErrors(cudaEventRecord(stop, 0));
            checkCudaErrors(cudaEventSynchronize(stop));
            CHECK_CUDA_ERROR();
            checkCudaErrors(cudaEventElapsedTime(&t, start, stop));
            t /= 1.e3;
            bdwth = ((double)globalWorkSize * maxRepeatsUnit * 16 * sizeof(float)) /
                            (t * 1000. * 1000. * 1000.);
            resultDB.AddResult("writeGlobalMemoryUnit", sizeStr, "GB/s", bdwth);
            resultDB.AddOverall("Bandwidth", "GB/s", bdwth);

            // Test 6
            checkCudaErrors(cudaEventRecord(start, 0));

            writeLocalMemory<<<blocks, threads>>>(d_mem2, numWordsFloat, maxRepeatsLocal);

            checkCudaErrors(cudaEventRecord(stop, 0));
            checkCudaErrors(cudaEventSynchronize(stop));
            CHECK_CUDA_ERROR();
            checkCudaErrors(cudaEventElapsedTime(&t, start, stop));
            t /= 1.e3;
            bdwth = ((double)globalWorkSize * maxRepeatsLocal * 16 * sizeof(float)) /
                            (t * 1000. * 1000. * 1000.);
            resultDB.AddResult("writeLocalMemory", sizeStr, "GB/s", bdwth);
        }
    }
    checkCudaErrors(cudaFree(d_mem1));
    checkCudaErrors(cudaFree(d_mem2));
    delete[] h_in;
    delete[] h_out;
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    TestTextureMem(resultDB, op, scalet);
}

// ****************************************************************************
// Function: TestTextureMem (from SHOC)
//
// Purpose:
//   Measures the bandwidth of texture memory for several access patterns
//   using a 2D texture including sequential, "random", and repeated access to
//   texture cache.  Texture memory is often a viable alternative to global
//   memory, especially when data access patterns prevent good coalescing.
//
// Arguments:
//   resultDB: results from the benchmark are stored to this resultd database
//   op: the options parser / parameter database
//   scalet: auto-scaling factor for the number of repetitions
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 11, 2009
//
// Modifications:
//   Gabriel Marin 06/09/2010: add auto-scaling factor
//
//   Jeremy Meredith, Tue Nov 23 13:45:54 EST 2010
//   Change data sizes to be larger, and textures to be 2D to match OpenCL
//   variant.  Dropped #iterations to compensate.  Had to remove validation
//   for now, which also matches the current OpenCL variant's behavior.
//
//   Jeremy Meredith, Wed Oct 10 11:54:32 EDT 2012
//   Kernel rep factor of 1024 on the last texture test on the biggest
//   texture size caused Windows to time out (the more-than-five-seconds-long
//   kernel problem).  I made kernel rep factor problem-size dependent.
//
// ****************************************************************************

void TestTextureMem(ResultDatabase &resultDB, OptionParser &op, double scalet) {
    // Enable quiet output
    bool quiet = op.getOptionBool("quiet");
    // Number of times to repeat each test
    const unsigned int passes = op.getOptionInt("passes");
    const bool uvm = op.getOptionBool("uvm");
    // Sizes of textures tested (in kb)
    const unsigned int nsizes = 5;
    const unsigned int sizes[] = {256, 512, 1024, 2048, 4096};
    // Number of texel accesses by each kernel
    const unsigned int kernelRepFactors[] = {2048, 2048, 2048, 2048, 512};
    // Number of times to repeat each kernel per test
    const unsigned int iterations = 1 * scalet;

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // make sure our texture behaves like we want....
    texA.normalized = false;
    texA.addressMode[0] = cudaAddressModeClamp;
    texA.addressMode[1] = cudaAddressModeClamp;
    texA.filterMode = cudaFilterModePoint;

    for (int j = 0; j < nsizes; j++) {
        if(!quiet) {
            cout << "Benchmarking Texture Memory, Test Size: " << j + 1 << " / 5\n";
        }
        const unsigned int size = 1024 * sizes[j];
        const unsigned int numFloat = size / sizeof(float);
        const unsigned int numFloat4 = size / sizeof(float4);
        size_t width, height;

        const unsigned int kernelRepFactor = kernelRepFactors[j];

        // Image memory sizes should be power of 2.
        size_t sizeLog = lround(log2(double(numFloat4)));
        height = 1 << (sizeLog >> 1); // height is the smaller size
        width = numFloat4 / height;

        const dim3 blockSize(16, 8);
        const dim3 gridSize(width / blockSize.x, height / blockSize.y);

        float *h_in = new float[numFloat];
        float *h_out = new float[numFloat4];
        float *d_out = NULL;
        if (uvm) {
            checkCudaErrors(cudaMallocManaged((void **)&d_out, numFloat4 * sizeof(float)));
        } else {
            checkCudaErrors(cudaMalloc((void **)&d_out, numFloat4 * sizeof(float)));
        }

        // Fill input data with some pattern
        for (unsigned int i = 0; i < numFloat; i++) {
            h_in[i] = (float)i;
            if (i < numFloat4) {
                h_out[i] = 0.0f;
            }
        }

        // Allocate a cuda array
        cudaArray *cuArray;
        checkCudaErrors(cudaMallocArray(&cuArray, &texA.channelDesc, width, height));

        // Copy in source data
        //checkCudaErrors(cudaMemcpyToArray(cuArray, 0, 0, h_in, size, cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, h_in, width, width, height, cudaMemcpyDefault));

        // Bind texture to the array
        checkCudaErrors(cudaBindTextureToArray(texA, cuArray));

        for (int p = 0; p < passes; p++) {
            // Test 1: Repeated Linear Access
            float t = 0.0f;

            checkCudaErrors(cudaEventRecord(start, 0));
            // read texels from texture
            for (int iter = 0; iter < iterations; iter++) {
                readTexels<<<gridSize, blockSize>>>(kernelRepFactor, d_out, width);
            }
            checkCudaErrors(cudaEventRecord(stop, 0));
            checkCudaErrors(cudaEventSynchronize(stop));
            CHECK_CUDA_ERROR();
            checkCudaErrors(cudaEventElapsedTime(&t, start, stop));
            t /= 1.e3;

            // Calculate speed in GB/s
            double speed = (double)kernelRepFactor * (double)iterations *
                                         (double)(size / (1000. * 1000. * 1000.)) / (t);

            char sizeStr[256];
            sprintf(sizeStr, "% 6dkB", size / 1024);
            resultDB.AddResult("TextureRepeatedLinearAccess", sizeStr, "GB/s",
                                                 speed);

            // Verify results
            checkCudaErrors(cudaMemcpy(h_out, d_out, numFloat4 * sizeof(float),
                                 cudaMemcpyDeviceToHost));

            // Test 2 Repeated Cache Access
            checkCudaErrors(cudaEventRecord(start, 0));
            for (int iter = 0; iter < iterations; iter++) {
                readTexelsInCache<<<gridSize, blockSize>>>(kernelRepFactor, d_out);
            }
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            CHECK_CUDA_ERROR();
            checkCudaErrors(cudaEventElapsedTime(&t, start, stop));
            t /= 1.e3;

            // Calculate speed in GB/s
            speed = (double)kernelRepFactor * (double)iterations *
                            ((double)size / (1000. * 1000. * 1000.)) / (t);

            sprintf(sizeStr, "% 6dkB", size / 1024);
            resultDB.AddResult("TextureRepeatedCacheHit", sizeStr, "GB/s", speed);

            // Verify results
            checkCudaErrors(cudaMemcpy(h_out, d_out, numFloat4 * sizeof(float),
                                 cudaMemcpyDeviceToHost));

            // Test 3 Repeated "Random" Access
            checkCudaErrors(cudaEventRecord(start, 0));

            // read texels from texture
            for (int iter = 0; iter < iterations; iter++) {
                readTexelsRandom<<<gridSize, blockSize>>>(kernelRepFactor, d_out, width, height);
            }

            checkCudaErrors(cudaEventRecord(stop, 0));
            checkCudaErrors(cudaEventSynchronize(stop));
            CHECK_CUDA_ERROR();
            checkCudaErrors(cudaEventElapsedTime(&t, start, stop));
            t /= 1.e3;

            // Calculate speed in GB/s
            speed = (double)kernelRepFactor * (double)iterations *
                            ((double)size / (1000. * 1000. * 1000.)) / (t);

            sprintf(sizeStr, "% 6dkB", size / 1024);
            resultDB.AddResult("TextureRepeatedRandomAccess", sizeStr, "GB/s",
                                                 speed);
        }
        delete[] h_in;
        delete[] h_out;
        checkCudaErrors(cudaFree(d_out));
        checkCudaErrors(cudaFreeArray(cuArray));
        checkCudaErrors(cudaUnbindTexture(texA));
    }
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
}

// Begin benchmark kernels

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads global memory coalesced. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. 
///             Add total threads into parameter for coalescing </remarks>
///
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size. </param>
/// <param name="repeat">	The repeat. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void readGlobalMemoryCoalesced(float *data, float *output, int globalWorkSize, int size, int repeat) {
    int gid = threadIdx.x + (blockDim.x * blockIdx.x), j = 0;
    float sum = 0;
    int s = gid;
    for (j = 0; j < repeat; ++j) {
        float a0 = data[(s + 0) & (size - 1)];
        float a1 = data[(s + globalWorkSize) & (size - 1)];
        float a2 = data[(s + globalWorkSize * 2) & (size - 1)];
        float a3 = data[(s + globalWorkSize * 3) & (size - 1)];
        float a4 = data[(s + globalWorkSize * 4) & (size - 1)];
        float a5 = data[(s + globalWorkSize * 5) & (size - 1)];
        float a6 = data[(s + globalWorkSize * 6) & (size - 1)];
        float a7 = data[(s + globalWorkSize * 7) & (size - 1)];
        float a8 = data[(s + globalWorkSize * 8) & (size - 1)];
        float a9 = data[(s + globalWorkSize * 9) & (size - 1)];
        float a10 = data[(s + globalWorkSize * 10) & (size - 1)];
        float a11 = data[(s + globalWorkSize * 11) & (size - 1)];
        float a12 = data[(s + globalWorkSize * 12) & (size - 1)];
        float a13 = data[(s + globalWorkSize * 13) & (size - 1)];
        float a14 = data[(s + globalWorkSize * 14) & (size - 1)];
        float a15 = data[(s + globalWorkSize * 15) & (size - 1)];
        sum += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 +
                     a13 + a14 + a15;
        s = (s + globalWorkSize * 16) & (size - 1);
    }
    output[gid] = sum;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads global memory unit by unit. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size. </param>
/// <param name="repeat">	The repeat. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void readGlobalMemoryUnit(float *data, float *output, int maxGroupSize, int size, int repeat) {
    int gid = threadIdx.x + (blockDim.x * blockIdx.x), j = 0;
    float sum = 0;
    int s = gid * maxGroupSize;
    for (j = 0; j < repeat; ++j) {
        float a0 = data[(s + 0) & (size - 1)];
        float a1 = data[(s + 1) & (size - 1)];
        float a2 = data[(s + 2) & (size - 1)];
        float a3 = data[(s + 3) & (size - 1)];
        float a4 = data[(s + 4) & (size - 1)];
        float a5 = data[(s + 5) & (size - 1)];
        float a6 = data[(s + 6) & (size - 1)];
        float a7 = data[(s + 7) & (size - 1)];
        float a8 = data[(s + 8) & (size - 1)];
        float a9 = data[(s + 9) & (size - 1)];
        float a10 = data[(s + 10) & (size - 1)];
        float a11 = data[(s + 11) & (size - 1)];
        float a12 = data[(s + 12) & (size - 1)];
        float a13 = data[(s + 13) & (size - 1)];
        float a14 = data[(s + 14) & (size - 1)];
        float a15 = data[(s + 15) & (size - 1)];
        sum += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 +
                     a13 + a14 + a15;
        s = (s + 16) & (size - 1);
    }
    output[gid] = sum;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads local memory. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="data">  	The data. </param>
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size. </param>
/// <param name="repeat">	The repeat. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void readLocalMemory(const float *data, float *output, int size, int repeat) {
    int gid = threadIdx.x + (blockDim.x * blockIdx.x), j = 0;
    float sum = 0;
    int tid = threadIdx.x, localSize = blockDim.x, grpid = blockIdx.x,
            litems = 2048 / localSize, goffset = localSize * grpid + tid * litems;
    int s = tid;
    __shared__ float lbuf[2048];
    for (; j < litems && j < (size - goffset); ++j)
        lbuf[tid * litems + j] = data[goffset + j];
    for (int i = 0; j < litems; ++j, ++i)
        lbuf[tid * litems + j] = data[i];
    __syncthreads();
    for (j = 0; j < repeat; ++j) {
        float a0 = lbuf[(s + 0) & (2047)];
        float a1 = lbuf[(s + 1) & (2047)];
        float a2 = lbuf[(s + 2) & (2047)];
        float a3 = lbuf[(s + 3) & (2047)];
        float a4 = lbuf[(s + 4) & (2047)];
        float a5 = lbuf[(s + 5) & (2047)];
        float a6 = lbuf[(s + 6) & (2047)];
        float a7 = lbuf[(s + 7) & (2047)];
        float a8 = lbuf[(s + 8) & (2047)];
        float a9 = lbuf[(s + 9) & (2047)];
        float a10 = lbuf[(s + 10) & (2047)];
        float a11 = lbuf[(s + 11) & (2047)];
        float a12 = lbuf[(s + 12) & (2047)];
        float a13 = lbuf[(s + 13) & (2047)];
        float a14 = lbuf[(s + 14) & (2047)];
        float a15 = lbuf[(s + 15) & (2047)];
        sum += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 +
                     a13 + a14 + a15;
        s = (s + 16) & (2047);
    }
    output[gid] = sum;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes to global memory coalesced. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size. </param>
/// <param name="repeat">	The repeat. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void writeGlobalMemoryCoalesced(float *output, int globalWorkSize, int size, int repeat) {
    int gid = threadIdx.x + (blockDim.x * blockIdx.x), j = 0;
    int s = gid;
    for (j = 0; j < repeat; ++j) {
        output[(s + 0) & (size - 1)] = gid;
        output[(s + globalWorkSize) & (size - 1)] = gid;
        output[(s + globalWorkSize * 2) & (size - 1)] = gid;
        output[(s + globalWorkSize * 3) & (size - 1)] = gid;
        output[(s + globalWorkSize * 4) & (size - 1)] = gid;
        output[(s + globalWorkSize * 5) & (size - 1)] = gid;
        output[(s + globalWorkSize * 6) & (size - 1)] = gid;
        output[(s + globalWorkSize * 7) & (size - 1)] = gid;
        output[(s + globalWorkSize * 8) & (size - 1)] = gid;
        output[(s + globalWorkSize * 9) & (size - 1)] = gid;
        output[(s + globalWorkSize * 10) & (size - 1)] = gid;
        output[(s + globalWorkSize * 11) & (size - 1)] = gid;
        output[(s + globalWorkSize * 12) & (size - 1)] = gid;
        output[(s + globalWorkSize * 13) & (size - 1)] = gid;
        output[(s + globalWorkSize * 14) & (size - 1)] = gid;
        output[(s + globalWorkSize * 15) & (size - 1)] = gid;
        s = (s + globalWorkSize * 16) & (size - 1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes to global memory unit. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size. </param>
/// <param name="repeat">	The repeat times of writing op. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void writeGlobalMemoryUnit(float *output, int maxGroupSize, int size, int repeat) {
    int gid = threadIdx.x + (blockDim.x * blockIdx.x), j = 0;
    int s = gid * maxGroupSize;
    for (j = 0; j < repeat; ++j) {
        output[(s + 0) & (size - 1)] = gid;
        output[(s + 1) & (size - 1)] = gid;
        output[(s + 2) & (size - 1)] = gid;
        output[(s + 3) & (size - 1)] = gid;
        output[(s + 4) & (size - 1)] = gid;
        output[(s + 5) & (size - 1)] = gid;
        output[(s + 6) & (size - 1)] = gid;
        output[(s + 7) & (size - 1)] = gid;
        output[(s + 8) & (size - 1)] = gid;
        output[(s + 9) & (size - 1)] = gid;
        output[(s + 10) & (size - 1)] = gid;
        output[(s + 11) & (size - 1)] = gid;
        output[(s + 12) & (size - 1)] = gid;
        output[(s + 13) & (size - 1)] = gid;
        output[(s + 14) & (size - 1)] = gid;
        output[(s + 15) & (size - 1)] = gid;
        s = (s + 16) & (size - 1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes to local memory. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="output">	[in,out] If non-null, the output. </param>
/// <param name="size">  	The size. </param>
/// <param name="repeat">	The number of writes. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void writeLocalMemory(float *output, int size, int repeat) {
    int gid = threadIdx.x + (blockDim.x * blockIdx.x), j = 0;
    int tid = threadIdx.x, localSize = blockDim.x, litems = 2048 / localSize;
    int s = tid;
    __shared__ float lbuf[2048];
    for (j = 0; j < repeat; ++j) {
        lbuf[(s + 0) & (2047)] = gid;
        lbuf[(s + 1) & (2047)] = gid;
        lbuf[(s + 2) & (2047)] = gid;
        lbuf[(s + 3) & (2047)] = gid;
        lbuf[(s + 4) & (2047)] = gid;
        lbuf[(s + 5) & (2047)] = gid;
        lbuf[(s + 6) & (2047)] = gid;
        lbuf[(s + 7) & (2047)] = gid;
        lbuf[(s + 8) & (2047)] = gid;
        lbuf[(s + 9) & (2047)] = gid;
        lbuf[(s + 10) & (2047)] = gid;
        lbuf[(s + 11) & (2047)] = gid;
        lbuf[(s + 12) & (2047)] = gid;
        lbuf[(s + 13) & (2047)] = gid;
        lbuf[(s + 14) & (2047)] = gid;
        lbuf[(s + 15) & (2047)] = gid;
        s = (s + 16) & (2047);
    }
    __syncthreads();
    for (j = 0; j < litems; ++j)
        output[gid] = lbuf[tid];
}

// Simple Repeated Linear Read from texture memory

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads the texels. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="n">		An int to process. </param>
/// <param name="d_out">	[in,out] If non-null, the out. </param>
/// <param name="width">	The width. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void readTexels(int n, float *d_out, int width) {
    int idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int idx_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int out_idx = idx_y * gridDim.x + idx_x;
    float sum = 0.0f;
    int width_bits = width - 1;
    for (int i = 0; i < n; i++) {
        float4 v = tex2D(texA, float(idx_x), float(idx_y));
        idx_x = (idx_x + 1) & width_bits;
        sum += v.x;
    }
    d_out[out_idx] = sum;
}

// Repeated read of only 4kb of texels (should fit in texture cache)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads texels in cache. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="n">		An int to process. </param>
/// <param name="d_out">	[in,out] If non-null, the out. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void readTexelsInCache(int n, float *d_out) {
    int idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int idx_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int out_idx = idx_y * gridDim.x + idx_x;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float4 v = tex2D(texA, float(idx_x), float(idx_y));
        sum += v.x;
    }
    d_out[out_idx] = sum;
}

// Read "random" texels

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads texels random. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="n">	 	An int to process. </param>
/// <param name="d_out"> 	[in,out] If non-null, the out. </param>
/// <param name="width"> 	The width. </param>
/// <param name="height">	The height. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void readTexelsRandom(int n, float *d_out, int width, int height) {
    int idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int idx_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int out_idx = idx_y * gridDim.x + idx_x;
    float sum = 0.0f;
    int width_bits = width - 1;
    int height_bits = height - 1;
    for (int i = 0; i < n; i++) {
        float4 v = tex2D(texA, float(idx_x), float(idx_y));
        idx_x = (idx_x * 3 + 29) & (width_bits);
        idx_y = (idx_y * 5 + 11) & (height_bits);
        sum += v.x;
    }
    d_out[out_idx] = sum;
}
