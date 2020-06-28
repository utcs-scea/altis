////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level0\busspeedreadback\BusSpeedReadback.cu
//
// summary:	Bus speed readback class
// 
// origin: SHOC Benchmark (https://github.com/vetter/shoc)
////////////////////////////////////////////////////////////////////////////////////////////////////

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
// Modifications: Bodun Hu (bodunhu@utexas.edu)
//   added safe call wrapper to CUDA call
// 
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
//   sizes across the bus to the host from the device and calculates the
//   bandwidth for each chunk size.
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
//    Added calculation of latency estimate.  //
// ****************************************************************************

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    cout << "Running BusSpeedReadback" << endl;

    const bool verbose = op.getOptionBool("verbose");
    const bool quiet = op.getOptionBool("quiet");
    const bool pinned = op.getOptionBool("pinned");

    // Sizes are in kb
    int nSizes = 20;
    int sizes[20] = {1,     2,     4,     8,      16,     32,    64,
                    128,   256,   512,   1024,   2048,   4096,  8192,
                    16384, 32768, 65536, 131072, 262144, 524288};
    long long numMaxFloats = 1024 * (sizes[nSizes - 1]) / 4;

    // Create some host memory pattern
    float *hostMem1 = NULL;
    float *hostMem2 = NULL;
    if (pinned) {
        cudaMallocHost((void **)&hostMem1, sizeof(float) * numMaxFloats);
        cudaError_t err1 = cudaGetLastError();
        cudaMallocHost((void **)&hostMem2, sizeof(float) * numMaxFloats);
        cudaError_t err2 = cudaGetLastError();
        while (err1 != cudaSuccess || err2 != cudaSuccess) {
        // free the first buffer if only the second failed
        if (err1 == cudaSuccess) {
            CUDA_SAFE_CALL(cudaFreeHost((void *)hostMem1));
        }

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
        cudaMallocHost((void **)&hostMem1, sizeof(float) * numMaxFloats);
        err1 = cudaGetLastError();
        cudaMallocHost((void **)&hostMem2, sizeof(float) * numMaxFloats);
        err2 = cudaGetLastError();
        }
    } else {
        hostMem1 = new float[numMaxFloats];
        hostMem2 = new float[numMaxFloats];
    }

    // fillup allocated region
    for (int i = 0; i < numMaxFloats; i++) {
        hostMem1[i] = i % 77;
    }

    float *device;
    cudaMalloc((void **)&device, sizeof(float) * numMaxFloats);
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
        cudaMalloc((void **)&device, sizeof(float) * numMaxFloats);
    }

    CUDA_SAFE_CALL(cudaMemcpy(device, hostMem1, numMaxFloats * sizeof(float),
                cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    const unsigned int passes = op.getOptionInt("passes");

    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

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

        CUDA_SAFE_CALL(cudaEventRecord(start, 0));
        CUDA_SAFE_CALL(cudaMemcpy(hostMem2, device, nbytes, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(stop));
        float t = 0;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&t, start, stop));
        // times[sizeIndex] = t;

        // Convert to GB/sec
        if (verbose && !quiet) {
            cout << "size " << sizes[sizeIndex] << "k took " << t << " ms\n";
        }

        double speed = (double(sizes[sizeIndex]) * 1024. / (1000 * 1000)) / t;
        resultDB.AddResult("ReadbackSpeed", "---", "GB/sec", speed);
        resultDB.AddOverall("ReadbackSpeed", "GB/sec", speed);
        }
    }

    // Cleanup
    CUDA_SAFE_CALL(cudaFree((void *)device));
    if (pinned) {
        CUDA_SAFE_CALL(cudaFreeHost((void *)hostMem1));
        CUDA_SAFE_CALL(cudaFreeHost((void *)hostMem2));
    } else {
        delete[] hostMem1;
        delete[] hostMem2;
        CUDA_SAFE_CALL(cudaEventDestroy(start));
        CUDA_SAFE_CALL(cudaEventDestroy(stop));
    }
}
