////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	\altis\src\cuda\level0\maxflops\MaxFlops.cu
//
// summary:	Maximum flops class
// 
// origin: SHOC (https://github.com/vetter/shoc)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OptionParser.h"
#include "ProgressBar.h"
#include "ResultDatabase.h"
#include "Utility.h"
#include "cudacommon.h"
#include "cuda_fp16.h"
#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Add kernels. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v">	 	Passed in operand. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void Add1(T *data, int nIters, T v);
template <class T> __global__ void Add2(T *data, int nIters, T v);
template <class T> __global__ void Add4(T *data, int nIters, T v);
template <class T> __global__ void Add8(T *data, int nIters, T v);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Multiply kernels. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v">	 	Passed in operand. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void Mul1(T *data, int nIters, T v);
template <class T> __global__ void Mul2(T *data, int nIters, T v);
template <class T> __global__ void Mul4(T *data, int nIters, T v);
template <class T> __global__ void Mul8(T *data, int nIters, T v);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	MAdd kernels. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v1">	 	The first value. </param>
/// <param name="v2">	 	The second value. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void MAdd1(T *data, int nIters, T v1, T v2);
template <class T> __global__ void MAdd2(T *data, int nIters, T v1, T v2);
template <class T> __global__ void MAdd4(T *data, int nIters, T v1, T v2);
template <class T> __global__ void MAdd8(T *data, int nIters, T v1, T v2);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	MulMAdd kernels. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v1">	 	The first value. </param>
/// <param name="v2">	 	The second value. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void MulMAdd1(T *data, int nIters, T v1, T v2);
template <class T> __global__ void MulMAdd2(T *data, int nIters, T v1, T v2);
template <class T> __global__ void MulMAdd4(T *data, int nIters, T v1, T v2);
template <class T> __global__ void MulMAdd8(T *data, int nIters, T v1, T v2);


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Forward Declarations
///  execute simple precision and double precision versions of the benchmarks. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="resultDB"> 	[in,out] The result database. </param>
/// <param name="npasses">  	The npasses. </param>
/// <param name="verbose">  	The verbose. </param>
/// <param name="quiet">		The quiet. </param>
/// <param name="repeatF">  	The repeat f. </param>
/// <param name="pb">			[in,out] The pb. </param>
/// <param name="precision">	The precision. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
void RunTest(ResultDatabase &resultDB, int npasses, int verbose, int quiet,
             float repeatF, ProgressBar &pb, const char *precision);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Block size to use in measurements. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE_SP 256
#define BLOCK_SIZE_DP 128

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Show testing progress bar. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="pb">	  	[in,out] The pb. </param>
/// <param name="verbose">	True to verbose. </param>
/// <param name="quiet">  	Whether to show the result. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void updateProgressBar(ProgressBar& pb, bool verbose, bool quiet) {
  pb.addItersDone();
  if (verbose || quiet) {
    return;
  }
  pb.Show(stdout);

}

// ****************************************************************************
// Function: addBenchmarkSpecOptions (from SHOC)
//
// Purpose:
//   Add benchmark specific options parsing
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
void addBenchmarkSpecOptions(OptionParser &op) {}

// ****************************************************************************
// Function: runBenchmark (from SHOC)
//
// Purpose:
//   This benchmark measures the max floating point capability of a GPU using
//   a highly unrolled kernel with a large number of floating point operations.
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
//    Jeremy Meredith, Fri May 14 11:23:10 EDT 2010
//    Made double precision a copy of SP, with a few tweaks.
//    Allow any capability at least 1.3 or 2.0 to use double.
//
//    Gabriel Marin, Thu Jan 13, 2010
//    Add the auto-generated kernels from the OpenCL implementation.
//    DP / SP implemented as templates for the new kernels.
//    Add text progress bar.
//    
//    Bodun Hu, January 1, 2019
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
  cout << "Running MaxFlops" << endl;
  bool verbose = op.getOptionBool("verbose");
  bool quiet = op.getOptionBool("quiet");
  const unsigned int passes = op.getOptionInt("passes");

  // Test to see if this device supports double precision
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  
  // Determine if compute capability supports double and half precision operations
  bool doDouble = false;
  bool doHalf = false;
  if ((deviceProp.major == 1 && deviceProp.minor >= 3) ||
      (deviceProp.major >= 2)) {
    doDouble = true;
  }
  if ((deviceProp.major == 5 && deviceProp.minor >= 3) ||
      (deviceProp.major >= 6)) {
    doHalf = true;
  }

  // TODO: maybe use device memory scaling solution
  // determine the speed of the device first. This determines the number of
  // iterations for all kernels.
  const unsigned int halfBufSize = 1024 * 1024;
  unsigned int halfNumFloats = halfBufSize / sizeof(float),
               numFloats = 2 * halfNumFloats;
  float *gpu_mem, *hostMem;
  hostMem = new float[numFloats];
  CUDA_SAFE_CALL(cudaMalloc((void **)&gpu_mem, halfBufSize * 2));
  // Initialize host data, with the first half the same as the second
  for (int j = 0; j < halfNumFloats; ++j) {
    hostMem[j] = hostMem[numFloats - j - 1] = (float)(drand48() * 10.0);
  }

  // Variables used for timing
  float t = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // copy host memory to GPU memory
  cudaEventRecord(start, 0); // do I even need this if I do not need the time?
  CUDA_SAFE_CALL(cudaMemcpy(gpu_mem, hostMem, halfBufSize * 2, cudaMemcpyHostToDevice));
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // Thread block configuration
  dim3 threads(BLOCK_SIZE_SP, 1, 1);
  dim3 blocks((numFloats) / BLOCK_SIZE_SP, 1, 1);

  // Decrease block size for devices with lower compute
  // capability.  Avoids an out of resources error
  if ((deviceProp.major == 1 && deviceProp.minor <= 2)) {
    threads.x = 128;
    blocks.x = (numFloats) / 128;
  }

  // Benchmark the MulMAdd2 kernel to compute a scaling factor.
  t = 0.0f;
  cudaEventRecord(start, 0);
  MulMAdd2<float><<<blocks, threads>>>(gpu_mem, 10, 3.75, 0.355);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  CHECK_CUDA_ERROR();
  cudaEventElapsedTime(&t, start, stop);
  t *= 1.e6;
  double repeatF = 1.1e07 / (double)t;
  fprintf(stdout, "Adjust repeat factor = %lg\n", repeatF);

  delete[] hostMem;
  CUDA_SAFE_CALL(cudaFree((void *)gpu_mem));

  // Initialize progress bar. We have 16 generic kernels and 2 hand tuned
  // kernels.
  // Each kernel is executed 'passes' number of times for each single precision
  // and
  // double precision (if avaialble).
  int totalRuns = 18 * passes;
  if (doDouble) {
    totalRuns += 18 * passes; // multiply by 2
  }
  if (doHalf) {
    totalRuns += 18 * passes; // multiply by 2
  }
  ProgressBar pb(totalRuns);

  // Run single precision kernels
  RunTest<float>(resultDB, passes, verbose, quiet, repeatF, pb, "SP-");

  // Run double precision kernels
  if (doDouble) {
    RunTest<double>(resultDB, passes, verbose, quiet, repeatF, pb, "DP-");
  }

  // Run half precision kernels
  if (doHalf) {
    RunTest<half>(resultDB, passes, verbose, quiet, repeatF, pb, "HP-");
  }

  CUDA_SAFE_CALL(cudaEventDestroy(start));
  CUDA_SAFE_CALL(cudaEventDestroy(stop));
}

template <class T>
bool checkResult(T *hostMem2, int numFloats, int halfNumFloats) {
    if(sizeof(T) == 2) {
        // CPU cannot do half precision functions
        return true;
    }
    // Check the result -- At a minimum the first half of memory
    // should match the second half exactly
    for (int j = 0; j < halfNumFloats; ++j) {
        if (hostMem2[j] != hostMem2[numFloats - j - 1]) {
            cerr << "Error; hostMem2[" << j << "]=" << (float)hostMem2[j]
                << " is different from its twin element hostMem2["
                << (numFloats - j - 1) << "]=" << (float)hostMem2[numFloats - j - 1]
                << "; stopping check\n";
            return false;
        }
    }
    return true;
}

template <class T>
void zeroOut(T *hostMem2, int numFloats) {
    // Zero out the test host memory
    for (int j = 0; j < numFloats; ++j) {
      hostMem2[j] = (T)0.0;
    }
}
// ****************************************************************************
// Function: RunTest (from SHOC)
//
// Purpose:
//   Template function used for specializing the generic kernels for
//   single precision and double precision.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//
// Returns:  nothing
//
// Programmer: Gabriel Marin
// Creation: January 13, 2010
//
// ****************************************************************************
template <class T>
void RunTest(ResultDatabase &resultDB, int npasses, int verbose, int quiet,
             float repeatF, ProgressBar &pb, const char *precision) {
  T *gpu_mem;
  char sizeStr[128];
  T *hostMem, *hostMem2;

  int realRepeats = (int)::round(repeatF * 20);
  if (realRepeats < 2) {
    realRepeats = 2;
  }

  // Alloc host memory
  int halfNumFloats = 1024 * 1024;
  int numFloats = 2 * halfNumFloats;
  hostMem = new T[numFloats];
  hostMem2 = new T[numFloats];

  CUDA_SAFE_CALL(cudaMalloc((void **)&gpu_mem, numFloats * sizeof(T)));

  // Variables used for timing
  float t = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Thread block configuration
  dim3 threads(128, 1, 1);
  dim3 blocks((numFloats) / 128, 1, 1);

  for (int pass = 0; pass < npasses; ++pass) {
    // Benchmark each generic kernel. Generate new random numbers for each run.
    ////////// Add1 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the Add1 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    Add1<T><<<blocks, threads>>>(gpu_mem, realRepeats, (T)10.0);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    double flopCount = (double)numFloats * 1 * realRepeats * 240 * 1;
    double gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("Add1"), sizeStr, "GFLOPS", gflop);
    resultDB.AddOverall("FLOPS", "", gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    if(!checkResult<T>(hostMem2, numFloats, halfNumFloats)) {
        break;
    }

    // update progress bar
    updateProgressBar(pb, verbose, quiet);

    ////////// Add2 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the Add2 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    Add2<T><<<blocks, threads>>>(gpu_mem, realRepeats, 10.0);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    flopCount = (double)numFloats * 1 * realRepeats * 120 * 2;
    gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("Add2"), sizeStr, "GFLOPS", gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    if(!checkResult<T>(hostMem2, numFloats, halfNumFloats)) {
        break;
    }

    // update progress bar
    updateProgressBar(pb, verbose, quiet);

    ////////// Add4 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the Add4 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    Add4<T><<<blocks, threads>>>(gpu_mem, realRepeats, (T)10.0);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    flopCount = (double)numFloats * 1 * realRepeats * 60 * 4;
    gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("Add4"), sizeStr, "GFLOPS", gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    if(!checkResult<T>(hostMem2, numFloats, halfNumFloats)) {
        break;
    }

    // update progress bar
    updateProgressBar(pb, verbose, quiet);

    ////////// Add8 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the Add8 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    Add8<T><<<blocks, threads>>>(gpu_mem, realRepeats, (T)10.0);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    flopCount = (double)numFloats * 1 * realRepeats * 30 * 8;
    gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("Add8"), sizeStr, "GFLOPS", gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    if(!checkResult<T>(hostMem2, numFloats, halfNumFloats)) {
        break;
    }

    // update progress bar
    updateProgressBar(pb, verbose, quiet);

    ////////// Mul1 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the Mul1 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    Mul1<T><<<blocks, threads>>>(gpu_mem, realRepeats, 1.01);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    flopCount = (double)numFloats * 2 * realRepeats * 200 * 1;
    gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("Mul1"), sizeStr, "GFLOPS", gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    if(!checkResult<T>(hostMem2, numFloats, halfNumFloats)) {
        break;
    }

    // update progress bar
    updateProgressBar(pb, verbose, quiet);

    ////////// Mul2 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the Mul2 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    Mul2<T><<<blocks, threads>>>(gpu_mem, realRepeats, 1.01);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    flopCount = (double)numFloats * 2 * realRepeats * 100 * 2;
    gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("Mul2"), sizeStr, "GFLOPS", gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    if(!checkResult<T>(hostMem2, numFloats, halfNumFloats)) {
        break;
    }

    // update progress bar
    updateProgressBar(pb, verbose, quiet);

    ////////// Mul4 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the Mul4 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    Mul4<T><<<blocks, threads>>>(gpu_mem, realRepeats, 1.01);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    flopCount = (double)numFloats * 2 * realRepeats * 50 * 4;
    gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("Mul4"), sizeStr, "GFLOPS", gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    if(!checkResult<T>(hostMem2, numFloats, halfNumFloats)) {
        break;
    }

    // update progress bar
    updateProgressBar(pb, verbose, quiet);

    ////////// Mul8 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the Mul8 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    Mul8<T><<<blocks, threads>>>(gpu_mem, realRepeats, 1.01);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    flopCount = (double)numFloats * 2 * realRepeats * 25 * 8;
    gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("Mul8"), sizeStr, "GFLOPS", gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    if(!checkResult<T>(hostMem2, numFloats, halfNumFloats)) {
        break;
    }

    // update progress bar
    updateProgressBar(pb, verbose, quiet);

    ////////// MAdd1 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the MAdd1 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    MAdd1<T><<<blocks, threads>>>(gpu_mem, realRepeats, 10.0, 0.9899);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    flopCount = (double)numFloats * 2 * realRepeats * 240 * 1;
    gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("MAdd1"), sizeStr, "GFLOPS", gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    checkResult<T>(hostMem2, numFloats, halfNumFloats);

    // update progress bar
    updateProgressBar(pb, verbose, quiet);

    ////////// MAdd2 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the MAdd2 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    MAdd2<T><<<blocks, threads>>>(gpu_mem, realRepeats, 10.0, 0.9899);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    flopCount = (double)numFloats * 2 * realRepeats * 120 * 2;
    gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("MAdd2"), sizeStr, "GFLOPS", gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    checkResult<T>(hostMem2, numFloats, halfNumFloats);

    // update progress bar
    updateProgressBar(pb, verbose, quiet);

    ////////// MAdd4 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the MAdd4 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    MAdd4<T><<<blocks, threads>>>(gpu_mem, realRepeats, 10.0, 0.9899);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    flopCount = (double)numFloats * 2 * realRepeats * 60 * 4;
    gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("MAdd4"), sizeStr, "GFLOPS", gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    checkResult<T>(hostMem2, numFloats, halfNumFloats);

    // update progress bar
    updateProgressBar(pb, verbose, quiet);

    ////////// MAdd8 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the MAdd8 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    MAdd8<T><<<blocks, threads>>>(gpu_mem, realRepeats, 10.0, 0.9899);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    flopCount = (double)numFloats * 2 * realRepeats * 30 * 8;
    gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("MAdd8"), sizeStr, "GFLOPS", gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    checkResult<T>(hostMem2, numFloats, halfNumFloats);

    // update progress bar
    updateProgressBar(pb, verbose, quiet);

    ////////// MulMAdd1 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the MulMAdd1 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    MulMAdd1<T><<<blocks, threads>>>(gpu_mem, realRepeats, 3.75, 0.355);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    flopCount = (double)numFloats * 3 * realRepeats * 160 * 1;
    gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("MulMAdd1"), sizeStr, "GFLOPS",
                       gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    checkResult<T>(hostMem2, numFloats, halfNumFloats);

    // update progress bar
    updateProgressBar(pb, verbose, quiet);

    ////////// MulMAdd2 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the MulMAdd2 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    MulMAdd2<T><<<blocks, threads>>>(gpu_mem, realRepeats, 3.75, 0.355);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    flopCount = (double)numFloats * 3 * realRepeats * 80 * 2;
    gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("MulMAdd2"), sizeStr, "GFLOPS",
                       gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    checkResult<T>(hostMem2, numFloats, halfNumFloats);

    // update progress bar
    updateProgressBar(pb, verbose, quiet);

    ////////// MulMAdd4 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the MulMAdd4 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    MulMAdd4<T><<<blocks, threads>>>(gpu_mem, realRepeats, 3.75, 0.355);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    flopCount = (double)numFloats * 3 * realRepeats * 40 * 4;
    gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("MulMAdd4"), sizeStr, "GFLOPS",
                       gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    checkResult<T>(hostMem2, numFloats, halfNumFloats);

    // update progress bar
    updateProgressBar(pb, verbose, quiet);

    ////////// MulMAdd8 //////////
    // Initialize host data, with the first half the same as the second
    for (int j = 0; j < halfNumFloats; ++j) {
      hostMem[j] = hostMem[numFloats - j - 1] = (T)(drand48() * 10.0);
    }

    // copy host memory to GPU memory
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(gpu_mem, hostMem, numFloats * sizeof(T), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Execute the MulMAdd8 kernel
    t = 0.0f;
    cudaEventRecord(start, 0);
    MulMAdd8<T><<<blocks, threads>>>(gpu_mem, realRepeats, 3.75, 0.355);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR();
    cudaEventElapsedTime(&t, start, stop);
    t *= 1.e6;

    // flopCount = numFloats(pixels) * flopCount/op * numLoopIters *
    // unrollFactor * numStreams
    flopCount = (double)numFloats * 3 * realRepeats * 20 * 8;
    gflop = flopCount / (double)(t);

    sprintf(sizeStr, "Size:%07d", numFloats);
    resultDB.AddResult(precision + string("MulMAdd8"), sizeStr, "GFLOPS",
                       gflop);

    zeroOut<T>(hostMem2, numFloats);

    // Read the result device memory back to the host
    cudaEventRecord(start, 0); // do I even need this if I do not need the time?
    cudaMemcpy(hostMem2, gpu_mem, numFloats * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    checkResult<T>(hostMem2, numFloats, halfNumFloats);

    // update progress bar
    updateProgressBar(pb, verbose, quiet);
  }

  delete[] hostMem;
  delete[] hostMem2;
  CUDA_SAFE_CALL(cudaFree((void *)gpu_mem));

  CUDA_SAFE_CALL(cudaEventDestroy(start));
  CUDA_SAFE_CALL(cudaEventDestroy(stop));
}

// Macros used to construct MaxFlops kernels
// Each mad OP is 32*2 = 64 FLOPS
#define OP                                                                     \
  {                                                                            \
    s0 = s6 * s5 + s28;                                                        \
    s1 = s7 * s6 + s29;                                                        \
    s2 = s8 * s7 + s30;                                                        \
    s3 = s9 * s8 + s31;                                                        \
    s4 = s10 * s9 + s0;                                                        \
    s5 = s11 * s10 + s1;                                                       \
    s6 = s12 * s11 + s2;                                                       \
    s7 = s13 * s12 + s3;                                                       \
    s8 = s14 * s13 + s4;                                                       \
    s9 = s15 * s14 + s5;                                                       \
    s10 = s16 * s15 + s6;                                                      \
    s11 = s17 * s16 + s7;                                                      \
    s12 = s18 * s17 + s8;                                                      \
    s13 = s19 * s18 + s9;                                                      \
    s14 = s20 * s19 + s10;                                                     \
    s15 = s21 * s20 + s11;                                                     \
    s16 = s22 * s21 + s12;                                                     \
    s17 = s23 * s22 + s13;                                                     \
    s18 = s24 * s23 + s14;                                                     \
    s19 = s25 * s24 + s15;                                                     \
    s20 = s26 * s25 + s16;                                                     \
    s21 = s27 * s26 + s17;                                                     \
    s22 = s28 * s27 + s18;                                                     \
    s23 = s29 * s28 + s19;                                                     \
    s24 = s30 * s29 + s20;                                                     \
    s25 = s31 * s30 + s21;                                                     \
    s26 = s0 * s31 + s22;                                                      \
    s27 = s1 * s0 + s23;                                                       \
    s28 = s2 * s1 + s24;                                                       \
    s29 = s3 * s2 + s25;                                                       \
    s30 = s4 * s3 + s26;                                                       \
    s31 = s5 * s4 + s27;                                                       \
  }

// so Each OP10 is 640 FLOPS
#define OP10                                                                   \
  { OP OP OP OP OP OP OP OP OP OP }

// Each mad+mul MMOP is 8*3 = 24 FLOPS
#define MMOP                                                                   \
  {                                                                            \
    s0 = s4 * s4 + s4;                                                         \
    s6 = s0 * s5;                                                              \
    s1 = s5 * s5 + s5;                                                         \
    s7 = s1 * s6;                                                              \
    s2 = s6 * s6 + s6;                                                         \
    s0 = s2 * s7;                                                              \
    s3 = s7 * s7 + s7;                                                         \
    s1 = s3 * s0;                                                              \
    s4 = s0 * s0 + s0;                                                         \
    s2 = s4 * s1;                                                              \
    s5 = s1 * s1 + s1;                                                         \
    s3 = s5 * s2;                                                              \
    s6 = s2 * s2 + s2;                                                         \
    s4 = s6 * s3;                                                              \
    s7 = s3 * s3 + s3;                                                         \
    s5 = s7 * s4;                                                              \
  }

// So each OP10 is 240 FLOPS
#define MMOP10                                                                 \
  { MMOP MMOP MMOP MMOP MMOP MMOP MMOP MMOP MMOP MMOP }

// v = 10.0
#define ADD1_OP s = v - s;
#define ADD2_OP ADD1_OP s2 = v - s2;
#define ADD4_OP                                                                \
  ADD2_OP s3 = v - s3;                                                         \
  s4 = v - s4;
#define ADD8_OP                                                                \
  ADD4_OP s5 = v - s5;                                                         \
  s6 = v - s6;                                                                 \
  s7 = v - s7;                                                                 \
  s8 = v - s8;

// v = 1.01
#define MUL1_OP s = s * s * v;
#define MUL2_OP MUL1_OP s2 = s2 * s2 * v;
#define MUL4_OP                                                                \
  MUL2_OP s3 = s3 * s3 * v;                                                    \
  s4 = s4 * s4 * v;
#define MUL8_OP                                                                \
  MUL4_OP s5 = s5 * s5 * v;                                                    \
  s6 = s6 * s6 * v;                                                            \
  s7 = s7 * s7 * v;                                                            \
  s8 = s8 * s8 * v;

// v1 = 10.0, v2 = 0.9899
#define MADD1_OP s = v1 - s * v2;
#define MADD2_OP MADD1_OP s2 = v1 - s2 * v2;
#define MADD4_OP                                                               \
  MADD2_OP s3 = v1 - s3 * v2;                                                  \
  s4 = v1 - s4 * v2;
#define MADD8_OP                                                               \
  MADD4_OP s5 = v1 - s5 * v2;                                                  \
  s6 = v1 - s6 * v2;                                                           \
  s7 = v1 - s7 * v2;                                                           \
  s8 = v1 - s8 * v2;

// v1 = 3.75, v2 = 0.355
#define MULMADD1_OP s = (v1 - v2 * s) * s;
#define MULMADD2_OP MULMADD1_OP s2 = (v1 - v2 * s2) * s2;
#define MULMADD4_OP                                                            \
  MULMADD2_OP s3 = (v1 - v2 * s3) * s3;                                        \
  s4 = (v1 - v2 * s4) * s4;
#define MULMADD8_OP                                                            \
  MULMADD4_OP s5 = (v1 - v2 * s5) * s5;                                        \
  s6 = (v1 - v2 * s6) * s6;                                                    \
  s7 = (v1 - v2 * s7) * s7;                                                    \
  s8 = (v1 - v2 * s8) * s8;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines simple add (minus) operation. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define ADD1_MOP20                                                             \
  ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP      \
      ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP  \
          ADD1_OP ADD1_OP
#define ADD2_MOP20                                                             \
  ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP      \
      ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP  \
          ADD2_OP ADD2_OP
#define ADD4_MOP10                                                             \
  ADD4_OP ADD4_OP ADD4_OP ADD4_OP ADD4_OP ADD4_OP ADD4_OP ADD4_OP ADD4_OP      \
      ADD4_OP
#define ADD8_MOP5 ADD8_OP ADD8_OP ADD8_OP ADD8_OP ADD8_OP

#define MUL1_MOP20                                                             \
  MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP      \
      MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP  \
          MUL1_OP MUL1_OP
#define MUL2_MOP20                                                             \
  MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP      \
      MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP  \
          MUL2_OP MUL2_OP
#define MUL4_MOP10                                                             \
  MUL4_OP MUL4_OP MUL4_OP MUL4_OP MUL4_OP MUL4_OP MUL4_OP MUL4_OP MUL4_OP      \
      MUL4_OP
#define MUL8_MOP5 MUL8_OP MUL8_OP MUL8_OP MUL8_OP MUL8_OP

#define MADD1_MOP20                                                            \
  MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP      \
      MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP  \
          MADD1_OP MADD1_OP MADD1_OP MADD1_OP
#define MADD2_MOP20                                                            \
  MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP      \
      MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP  \
          MADD2_OP MADD2_OP MADD2_OP MADD2_OP
#define MADD4_MOP10                                                            \
  MADD4_OP MADD4_OP MADD4_OP MADD4_OP MADD4_OP MADD4_OP MADD4_OP MADD4_OP      \
      MADD4_OP MADD4_OP
#define MADD8_MOP5 MADD8_OP MADD8_OP MADD8_OP MADD8_OP MADD8_OP

#define MULMADD1_MOP20                                                         \
  MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP      \
      MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP  \
          MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP          \
              MULMADD1_OP MULMADD1_OP MULMADD1_OP
#define MULMADD2_MOP20                                                         \
  MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP      \
      MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP  \
          MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP          \
              MULMADD2_OP MULMADD2_OP MULMADD2_OP
#define MULMADD4_MOP10                                                         \
  MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP      \
      MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP
#define MULMADD8_MOP5                                                          \
  MULMADD8_OP MULMADD8_OP MULMADD8_OP MULMADD8_OP MULMADD8_OP

// Construct ADD_MOP operations
template <class T> __global__ void Add1(T *data, int nIters, T v) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid];
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 20 operations.
       Unroll 12 more times for 240 operations total.
     */
    ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20
        ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20 ADD1_MOP20
  }
  data[gid] = s;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Two operands copmutation. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The number of iters. </param>
/// <param name="v">	 	operand. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void Add2(T *data, int nIters, T v) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid], s2 = (T)10.0f - s;
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 20 operations.
       Unroll 6 more times for 120 operations total.
     */
    ADD2_MOP20 ADD2_MOP20 ADD2_MOP20 ADD2_MOP20 ADD2_MOP20 ADD2_MOP20
  }
  data[gid] = s + s2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Four operands operation. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The number of iters. </param>
/// <param name="v">	 	value operand. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void Add4(T *data, int nIters, T v) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid], s2 = (T)10.0f - s, s3 = (T)9.0f - s, s4 = (T)9.0f - s2;
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 10 operations.
       Unroll 6 more times for 60 operations total.
     */
    ADD4_MOP10 ADD4_MOP10 ADD4_MOP10 ADD4_MOP10 ADD4_MOP10 ADD4_MOP10
  }
  data[gid] = (s + s2) + (s3 + s4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Eight operands op. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v">	 	operand. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void Add8(T *data, int nIters, T v) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid], s2 = (T)10.0f - s, s3 = (T)9.0f - s, s4 = (T)9.0f - s2,
             s5 = (T)8.0f - s, s6 = (T)8.0f - s2, s7 = (T)7.0f - s, s8 = (T)7.0f - s2;
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 5 operations.
       Unroll 6 more times for 30 operations total.
     */
    ADD8_MOP5 ADD8_MOP5 ADD8_MOP5 ADD8_MOP5 ADD8_MOP5 ADD8_MOP5
  }
  data[gid] = ((s + s2) + (s3 + s4)) + ((s5 + s6) + (s7 + s8));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	One operand multiply. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v">	 	operand. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void Mul1(T *data, int nIters, T v) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid] - data[gid] + (T)0.999f;
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 20 operations.
       Unroll 10 more times for 200 operations total.
     */
    MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20 MUL1_MOP20
        MUL1_MOP20 MUL1_MOP20 MUL1_MOP20
  }
  data[gid] = s;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Two operand multiply. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v">	 	operand. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void Mul2(T *data, int nIters, T v) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid] - data[gid] + (T)0.999f, s2 = s - (T)0.0001f;
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 20 operations.
       Unroll 5 more times for 100 operations total.
     */
    MUL2_MOP20 MUL2_MOP20 MUL2_MOP20 MUL2_MOP20 MUL2_MOP20
  }
  data[gid] = s + s2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Four operand multiply. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v">	 	operand. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void Mul4(T *data, int nIters, T v) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid] - data[gid] + (T)0.999f, s2 = s - (T)0.0001f,
             s3 = s - (T)0.0002f, s4 = s - (T)0.0003f;
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 10 operations.
       Unroll 5 more times for 50 operations total.
     */
    MUL4_MOP10 MUL4_MOP10 MUL4_MOP10 MUL4_MOP10 MUL4_MOP10
  }
  data[gid] = (s + s2) + (s3 + s4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Eight operand multiply. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v">	 	operand to be applied. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void Mul8(T *data, int nIters, T v) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid] - data[gid] + (T)0.999f, s2 = s - (T)0.0001f,
             s3 = s - (T)0.0002f, s4 = s - (T)0.0003f, s5 = s - (T)0.0004f,
             s6 = s - (T)0.0005f, s7 = s - (T)0.0006f, s8 = s - (T)0.0007f;
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 5 operations.
       Unroll 5 more times for 25 operations total.
     */
    MUL8_MOP5 MUL8_MOP5 MUL8_MOP5 MUL8_MOP5 MUL8_MOP5
  }
  data[gid] = ((s + s2) + (s3 + s4)) + ((s5 + s6) + (s7 + s8));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Multiply and then add </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v1">	 	The first value. </param>
/// <param name="v2">	 	The second value. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void MAdd1(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid];
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 20 operations.
       Unroll 12 more times for 240 operations total.
     */
    MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
        MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20 MADD1_MOP20
  }
  data[gid] = s;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Two multiply and then add operation. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v1">	 	The first value. </param>
/// <param name="v2">	 	The second value. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void MAdd2(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid], s2 = (T)10.0f - s;
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 20 operations.
       Unroll 6 more times for 120 operations total.
     */
    MADD2_MOP20 MADD2_MOP20 MADD2_MOP20 MADD2_MOP20 MADD2_MOP20 MADD2_MOP20
  }
  data[gid] = s + s2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Four multiply and then add. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v1">	 	The first value. </param>
/// <param name="v2">	 	The second value. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void MAdd4(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid], s2 = (T)10.0f - s, s3 = (T)9.0f - s, s4 = (T)9.0f - s2;
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 10 operations.
       Unroll 6 more times for 60 operations total.
     */
    MADD4_MOP10 MADD4_MOP10 MADD4_MOP10 MADD4_MOP10 MADD4_MOP10 MADD4_MOP10
  }
  data[gid] = (s + s2) + (s3 + s4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Eight multiply and then add. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v1">	 	The first value. </param>
/// <param name="v2">	 	The second value. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void MAdd8(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid], s2 = (T)10.0f - s, s3 = (T)9.0f - s, s4 = (T)9.0f - s2,
             s5 = (T)8.0f - s, s6 = (T)8.0f - s2, s7 = (T)7.0f - s, s8 = (T)7.0f - s2;
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 5 operations.
       Unroll 6 more times for 30 operations total.
     */
    MADD8_MOP5 MADD8_MOP5 MADD8_MOP5 MADD8_MOP5 MADD8_MOP5 MADD8_MOP5
  }
  data[gid] = ((s + s2) + (s3 + s4)) + ((s5 + s6) + (s7 + s8));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	One multiply, add, then multiply operation. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v1">	 	The first value. </param>
/// <param name="v2">	 	The second value. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void MulMAdd1(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid];
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 20 operations.
       Unroll 8 more times for 160 operations total.
     */
    MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20
        MULMADD1_MOP20 MULMADD1_MOP20 MULMADD1_MOP20
  }
  data[gid] = s;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Two multiply, add, then multiply operation. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v1">	 	The first value. </param>
/// <param name="v2">	 	The second value. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void MulMAdd2(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid], s2 = (T)10.0f - s;
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 20 operations.
       Unroll 4 more times for 80 operations total.
     */
    MULMADD2_MOP20 MULMADD2_MOP20 MULMADD2_MOP20 MULMADD2_MOP20
  }
  data[gid] = s + s2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Four multiply, add, then multiply operation. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v1">	 	The first value. </param>
/// <param name="v2">	 	The second value. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void MulMAdd4(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid], s2 = (T)10.0f - s, s3 = (T)9.0f - s, s4 = (T)9.0f - s2;
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 10 operations.
       Unroll 4 more times for 40 operations total.
     */
    MULMADD4_MOP10 MULMADD4_MOP10 MULMADD4_MOP10 MULMADD4_MOP10
  }
  data[gid] = (s + s2) + (s3 + s4);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Eight Four multiply, add, then multiply operation. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="data">  	[in,out] If non-null, the data. </param>
/// <param name="nIters">	The iters. </param>
/// <param name="v1">	 	The first value. </param>
/// <param name="v2">	 	The second value. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> __global__ void MulMAdd8(T *data, int nIters, T v1, T v2) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  register T s = data[gid], s2 = (T)10.0f - s, s3 = (T)9.0f - s, s4 = (T)9.0f - s2,
             s5 = (T)8.0f - s, s6 = (T)8.0f - s2, s7 = (T)7.0f - s, s8 = (T)7.0f - s2;
  for (int j = 0; j < nIters; ++j) {
    /* Each macro op has 5 operations.
       Unroll 4 more times for 20 operations total.
     */
    MULMADD8_MOP5 MULMADD8_MOP5 MULMADD8_MOP5 MULMADD8_MOP5
  }
  /// <summary>	. </summary>
  data[gid] = ((s + s2) + (s3 + s4)) + ((s5 + s6) + (s7 + s8));
}
