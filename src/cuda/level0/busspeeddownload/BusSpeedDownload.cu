#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include <stdio.h>

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific command line argument parsing.
//
//   -nopinned
//   This option controls whether page-locked or "pinned" memory is used.
//   The use of pinned memory typically results in higher bandwidth for data
//   transfer between host and device.
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: September 08, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("pinned", OPT_BOOL, "1", "use pinned (pagelocked) memory");
}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   Measures the bandwidth of the bus connecting the host processor to the
//   OpenCL device.  This benchmark repeatedly transfers data chunks of various
//   sizes across the bus to the OpenCL device, and calculates the bandwidth.
//
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: September 08, 2009
//
// Modifications:
//    Jeremy Meredith, Wed Dec  1 17:05:27 EST 2010
//    Added calculation of latency estimate.
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
  cout << "Running BusSpeedDownload" << endl;
  const bool verbose = op.getOptionBool("verbose");
  const bool quiet = op.getOptionBool("quiet");
  const bool pinned = op.getOptionBool("pinned");
  const int id = op.getOptionInt("device");

  // Sizes are in kb
  int nSizes = 20;
  int sizes[20] = {1,     2,     4,     8,      16,     32,    64,
                   128,   256,   512,   1024,   2048,   4096,  8192,
                   16384, 32768, 65536, 131072, 262144, 524288};
  long long numMaxFloats = 1024 * (sizes[nSizes - 1]) / 4;

  // Create some host memory pattern
  float *hostMem = NULL;
  if (pinned) {
#ifdef UNIFIED_MEMORY
    CUDA_SAFE_CALL(cudaMallocManaged((void **)&hostMem, sizeof(float) * numMaxFloats));
    CUDA_SAFE_CALL(cudaMemPrefetchAsync(hostMem, sizeof(float) * numMaxFloats, cudaCpuDeviceId));
#else
    cudaMallocHost((void **)&hostMem, sizeof(float) * numMaxFloats);
#endif
    while (cudaGetLastError() != cudaSuccess) {
      // drop the size and try again
      if (verbose && !quiet) {
        cout << " - dropping size allocating pinned mem\n";
      }
      --nSizes;
      if (nSizes < 1) {
        cerr << "Error: Couldn't allocated any pinned buffer\n";
        return;
      }
      numMaxFloats = 1024 * (sizes[nSizes - 1]) / 4;
#ifdef UNIFIED_MEMORY
    CUDA_SAFE_CALL(cudaMallocManaged((void **)&hostMem, sizeof(float) * numMaxFloats));
    CUDA_SAFE_CALL(cudaMemPrefetchAsync(hostMem, sizeof(float) * numMaxFloats, cudaCpuDeviceId));
#else
      cudaMallocHost((void **)&hostMem, sizeof(float) * numMaxFloats);
#endif
    }
  } else {
    hostMem = new float[numMaxFloats];
  }

  for (int i = 0; i < numMaxFloats; i++) {
    hostMem[i] = i % 77;
  }

  float *device;
#ifdef UNIFIED_MEMORY
  if (pinned)
    device = hostMem;
  else {
    cudaMallocManaged((void **)&device, sizeof(float) * numMaxFloats);
    CUDA_SAFE_CALL(cudaMemPrefetchAsync(device, sizeof(float) * numMaxFloats, id));
  }
#else
  cudaMalloc((void **)&device, sizeof(float) * numMaxFloats);
#endif
  while (cudaGetLastError() != cudaSuccess) {
    // drop the size and try again
    if (verbose && !quiet) {
      cout << " - dropping size allocating device mem\n";
    }
    --nSizes;
    if (nSizes < 1) {
      cerr << "Error: Couldn't allocated any device buffer\n";
      return;
    }
    numMaxFloats = 1024 * (sizes[nSizes - 1]) / 4;
#ifdef UNIFIED_MEMORY
    if (pinned)
        device = hostMem;
    else {
        cudaMallocManaged((void **)&device, sizeof(float) * numMaxFloats);
        CUDA_SAFE_CALL(cudaMemPrefetchAsync(device, sizeof(float) * numMaxFloats, id));
    }
#else
    cudaMalloc((void **)&device, sizeof(float) * numMaxFloats);
#endif
  }

  const unsigned int passes = op.getOptionInt("passes");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Three passes, forward and backward both
  for (int pass = 0; pass < passes; pass++) {
    // store the times temporarily to estimate latency
    // float times[nSizes];
    // Step through sizes forward on even passes and backward on odd
    for (int i = 0; i < nSizes; i++) {
      int sizeIndex;
      if ((pass % 2) == 0)
        sizeIndex = i;
      else
        sizeIndex = (nSizes - 1) - i;

      int nbytes = sizes[sizeIndex] * 1024;

      cudaEventRecord(start, 0);
#ifdef UNIFIED_MEMORY
      if (pinned)
        CUDA_SAFE_CALL(cudaMemPrefetchAsync(device, nbytes, id));
      else
        CUDA_SAFE_CALL(cudaMemcpy(device, hostMem, nbytes, cudaMemcpyHostToDevice));

      cudaDeviceSynchronize();

      // wait 
#else
      CUDA_SAFE_CALL(cudaMemcpy(device, hostMem, nbytes, cudaMemcpyHostToDevice));
#endif
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      float t = 0;
      cudaEventElapsedTime(&t, start, stop);
      // times[sizeIndex] = t;

      // Convert to GB/sec
      if (verbose && !quiet) {
        cout << "size " << sizes[sizeIndex] << "k took " << t << " ms\n";
      }

      double speed = (double(sizes[sizeIndex]) * 1024. / (1000 * 1000)) / t;
      resultDB.AddResult("DownloadSpeed", "---", "GB/sec", speed);
      resultDB.AddOverall("DownloadSpeed", "GB/sec", speed);
    }
  }

  // Cleanup
  CUDA_SAFE_CALL(cudaFree((void *)device));
  if (pinned) {
#ifndef UNIFIED_MEMORY
    CUDA_SAFE_CALL(cudaFreeHost((void *)hostMem));
#endif
  } else {
    delete[] hostMem;
  }
  CUDA_SAFE_CALL(cudaEventDestroy(start));
  CUDA_SAFE_CALL(cudaEventDestroy(stop));
}
