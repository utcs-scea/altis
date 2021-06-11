/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* This sample queries the properties of the CUDA devices present in the system
 * via CUDA Runtime API. */

/*
 * Modification: Bodun Hu. 06/11/2021
 * Generate C/C++ header file with device-specific parameters.
 */

// std::system includes

#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <string>

int *pArgc = NULL;
char **pArgv = NULL;

#if CUDART_VERSION < 5000

// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
                             int device) {
  CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

  if (CUDA_SUCCESS != error) {
    fprintf(
        stderr,
        "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
        error, __FILE__, __LINE__);

    exit(EXIT_FAILURE);
  }
}

#endif /* CUDART_VERSION < 5000 */

static inline void writeSharedMem (std::fstream &cudaHdrFile, cudaDeviceProp &deviceProp) {
    size_t default_bank_width = 4; // in bytes

    /* write total shared mem per block */
    cudaHdrFile << "#define SHARED_MEM_PER_BLOCK    " << 
        deviceProp.sharedMemPerBlock << std::endl;


    /* share mem per sm */
    cudaHdrFile << "#define SHARED_MEM_PER_SM    " <<
        deviceProp.sharedMemPerMultiprocessor << std::endl;

    int sm_major_capability = deviceProp.major;
    int sm_minor_capability = deviceProp.minor;
    switch (sm_major_capability)
    {
        case 1:
        {
            cudaHdrFile << "#define SHARED_MEMORY_BANKS    " << 16 << std::endl;
            cudaHdrFile << "#define SHARED_MEMORY_BANK_BANDWIDTH    " << 4 << " // Each bank has a bandwidth of 32 bits per clock cycle" << std::endl;
            break;
        }
        case 2:
        {
            cudaHdrFile << "#define SHARED_MEMORY_BANKS    " << 32 << std::endl;
            cudaHdrFile << "#define SHARED_MEMORY_BANK_BANDWIDTH    " << 4 << " // Each bank has a bandwidth of 32 bits per two clock cycle" << std::endl;
            break;
        }
        case 3:
        {
            cudaHdrFile << "#define SHARED_MEMORY_BANKS    " << 32 << std::endl;
            cudaHdrFile << "#define SHARED_MEMORY_BANK_BANDWIDTH    " << 8 << " // Each bank has a bandwidth of 64 bits per clock cycle, consider using cudaDeviceSetSharedMemConfig()" << std::endl;
            break;
        }
        /* Technically there is not device of cap 4.X */
        case 4:
        {
            cudaHdrFile << "#define SHARED_MEMORY_BANKS    " << 32 << std::endl;
            cudaHdrFile << "#define SHARED_MEMORY_BANK_BANDWIDTH    " << 4 << " // Each bank has a bandwidth of 32 bits per clock cycle" << std::endl;
            break;
        }
        /*
         * Based on https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks,
         * On devices of compute capability 5.x or newer, each bank has a bandwidth
         * of 32 bits every clock cycle, and successive 32-bit words are assigned
         * to successive banks
         * */
        case 5:
        {
            cudaHdrFile << "#define SHARED_MEMORY_BANKS    " << 32 << std::endl;
            cudaHdrFile << "#define SHARED_MEMORY_BANK_BANDWIDTH    " << 4 << " // Each bank has a bandwidth of 32 bits per clock cycle" << std::endl;
            break;
        }
        case 6:
        {
            cudaHdrFile << "#define SHARED_MEMORY_BANKS    " << 32 << std::endl;
            cudaHdrFile << "#define SHARED_MEMORY_BANK_BANDWIDTH    " << 4 << " // Each bank has a bandwidth of 32 bits per clock cycle" << std::endl;
            break;
        }
        case 7:
        {
            cudaHdrFile << "#define SHARED_MEMORY_BANKS    " << 32 << std::endl;
            cudaHdrFile << "#define SHARED_MEMORY_BANK_BANDWIDTH    " << 4 << " // Each bank has a bandwidth of 32 bits per clock cycle (no doc)" << std::endl;
            break;
        }
        case 8:
        {
            cudaHdrFile << "#define SHARED_MEMORY_BANKS    " << 32 << std::endl;
            cudaHdrFile << "#define SHARED_MEMORY_BANK_BANDWIDTH    " << 4 << " // Each bank has a bandwidth of 32 bits per clock cycle (no doc)" << std::endl;
            break;
        }
        default:
        {
            cudaHdrFile << "#define SHARED_MEMORY_BANKS    " << 32 << std::endl;
            cudaHdrFile << "#define SHARED_MEMORY_BANK_BANDWIDTH    " << 4 << " // Each bank has a bandwidth of 32 bits per clock cycle (no doc)" << std::endl;

            //std::cerr << "No matching CUDA device capability." << std::endl;
            //exit(1);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  pArgc = &argc;
  pArgv = argv;

  printf("%s Starting...\n\n", argv[0]);
  std::string hdrPath = std::string(argv[1]);
  std::string filename ("cuda_device_attr.h");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  std::string fullPath = hdrPath.append("\\").append(filename);
#else
  std::string fullPath = hdrPath.append("/").append(filename);
#endif

  if (!argv[2]) {
    std::cerr << "Please specify device to querry" << std::endl;
    exit(0);
  }
  int deviceToQuerry = std::stoi(std::string(argv[2]));

  /* Starts writting file */
  std::fstream cudaHdrFile;
  cudaHdrFile.open(hdrPath, std::fstream::out | std::fstream::trunc);

  if (!cudaHdrFile) {
      std::cerr << "Invalid file Path given, exiting..." << std::endl;
      exit(-1);
  }

  int dev, driverVersion = 0, runtimeVersion = 0;

    /* Initialize device */
    if (gpuDeviceInit(deviceToQuerry) < 0) {
        std::cerr << "Invalid device query ID, exiting..." << std::endl;
        exit(-1);
    }

    dev = deviceToQuerry;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    /* write comment */
    cudaHdrFile << "/* This header file is used for " << deviceProp.name << ". */" << std::endl;

    /* Write header guards */
    cudaHdrFile << "#ifndef __CUDA_DEVICE_ATTR_H__" << std::endl;
    cudaHdrFile << "#define __CUDA_DEVICE_ATTR_H__" << std::endl << std::endl;


    printf(
      " CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
           static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);

        /* define number of cuda capable devices */
        cudaHdrFile << "#define CUDA_DEVICE_NUM    " << deviceCount << std::endl;
    }

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    /* write device ID and its name */
    cudaHdrFile << "#define CUDA_DEVICE_ID    " << dev << std::endl;
    cudaHdrFile << "#define CUDA_DEVICE_NAME    " << "\"" << deviceProp.name << "\"" << std::endl;

    // Console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    /* write device CUDA driver verion */
    cudaHdrFile << "#define CUDA_MAJOR_VERSION    " << driverVersion / 1000 << std::endl;
    cudaHdrFile << "#define CUDA_MINOR_VERSION    " << (driverVersion % 100) / 10 << std::endl;

    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);

    /* write CUDA capability */
    cudaHdrFile << "#define CUDA_MAJOR_CAPABILITY    " << deviceProp.major << std::endl;
    cudaHdrFile << "#define CUDA_MINOR_CAPABILITY    " << deviceProp.minor << std::endl;


    char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(msg, sizeof(msg),
             "  Total amount of global memory:                 %.0f MBytes "
             "(%llu bytes)\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
             (unsigned long long)deviceProp.totalGlobalMem);
    /*write total amount of global mem */
    cudaHdrFile << "#define GLOBAL_MEM    " << (unsigned long long)deviceProp.totalGlobalMem << std::endl;

#else
    snprintf(msg, sizeof(msg),
             "  Total amount of global memory:                 %.0f MBytes "
             "(%llu bytes)\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
             (unsigned long long)deviceProp.totalGlobalMem);

    /* write total amount of global mem */
    cudaHdrFile << "#define GLOBAL_MEM    " << (unsigned long long)deviceProp.totalGlobalMem << std::endl;

#endif
    printf("%s", msg);

    printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
           deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount);

    /* write hardware info */
    cudaHdrFile << "#define SM_COUNT    " << deviceProp.multiProcessorCount << std::endl;
    cudaHdrFile << "#define CUDA_CORES_PER_SM    " << 
        _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) << std::endl;
    cudaHdrFile << "#define CUDA_CORES    " << 
        _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * 
        deviceProp.multiProcessorCount << std::endl;


    printf(
        "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
        "GHz)\n",
        deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
    // This is supported in CUDA 5.0 (runtime API device properties)
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n",
           deviceProp.memoryBusWidth);

    /* write memory bus width */
    cudaHdrFile << "#define MEM_BUS_WIDTH    " << deviceProp.memoryBusWidth << std::endl;

    if (deviceProp.l2CacheSize) {
      printf("  L2 Cache Size:                                 %d bytes\n",
             deviceProp.l2CacheSize);

      /* write L2 cache size */
    cudaHdrFile << "#define L2_CACHE_SIZE    " << deviceProp.l2CacheSize << std::endl;
    }

#else
    // This only available in CUDA 4.0-4.2 (but these were only exposed in the
    // CUDA Driver API)
    int memoryClock;
    getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                          dev);
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           memoryClock * 1e-3f);
    int memBusWidth;
    getCudaAttribute<int>(&memBusWidth,
                          CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
    printf("  Memory Bus Width:                              %d-bit\n",
           memBusWidth);
    
    /* write memory bus width */
    cudaHdrFile << "#define MEM_BUS_WIDTH    " << memBusWidth << std::endl;

    int L2CacheSize;
    getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

    if (L2CacheSize) {
      printf("  L2 Cache Size:                                 %d bytes\n",
             L2CacheSize);

      /* write l2 cache size */
        cudaHdrFile << "#define L2_CACHE_SIZE    " << L2CacheSize << std::endl;

    }

#endif

    printf(
        "  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
        "%d), 3D=(%d, %d, %d)\n",
        deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
        deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
        deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);

    /* write max texture dim */
    cudaHdrFile << "#define MAX_TEXTURE_1D_DIM    " << deviceProp.maxTexture1D << std::endl;
    cudaHdrFile << "#define MAX_TEXTURE_2D_X    " << deviceProp.maxTexture2D[0] << std::endl;
    cudaHdrFile << "#define MAX_TEXTURE_2D_Y    " << deviceProp.maxTexture2D[1] << std::endl;
    cudaHdrFile << "#define MAX_TEXTURE_3D_X    " << deviceProp.maxTexture3D[0] << std::endl;
    cudaHdrFile << "#define MAX_TEXTURE_3D_Y    " << deviceProp.maxTexture3D[1] << std::endl;
    cudaHdrFile << "#define MAX_TEXTURE_3D_Z    " << deviceProp.maxTexture3D[2] << std::endl;



    printf(
        "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
        deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);

    /* write layered 1d tex size */
    cudaHdrFile << "#define MAX_LAYERED_1D_TEXTURE_SIZE    " << 
        deviceProp.maxTexture1DLayered[0] << std::endl;
    cudaHdrFile << "#define MAX_LAYERED_1D_TEXTURE_LAYERS    " << 
        deviceProp.maxTexture1DLayered[1] << std::endl;

    printf(
        "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
        "layers\n",
        deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
        deviceProp.maxTexture2DLayered[2]);

    /* write layered 2d tex size */
    cudaHdrFile << "#define MAX_LAYERED_2D_TEXTURE_SIZE_X    " 
        << deviceProp.maxTexture2DLayered[0] << std::endl;
    cudaHdrFile << "#define MAX_TEXTURE_2D_TEXTURE_SIZE_Y    " 
        << deviceProp.maxTexture2DLayered[1] << std::endl;
    cudaHdrFile << "#define MAX_TEXTURE_2D_TEXTURE_LAYERS    "
        << deviceProp.maxTexture2DLayered[2] << std::endl;


    printf("  Total amount of constant memory:               %zu bytes\n",
           deviceProp.totalConstMem);

    /* write max const mem */
    cudaHdrFile << "#define CONST_MEM    " << deviceProp.totalConstMem << std::endl;


    writeSharedMem(cudaHdrFile, deviceProp);

    printf("  Total amount of shared memory per block:       %zu bytes\n",
           deviceProp.sharedMemPerBlock);

    
    printf("  Total shared memory per multiprocessor:        %zu bytes\n",
           deviceProp.sharedMemPerMultiprocessor);

    
    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);

    /* total number of regs per block */
    cudaHdrFile << "#define REGS_PER_BLOCK    " <<
        deviceProp.regsPerBlock << std::endl;


    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);

    /* warp size */
    cudaHdrFile << "#define WARP_SIZE    " << deviceProp.warpSize << std::endl;


    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);

    /* max number of threads per SM */
    cudaHdrFile << "#define MAX_THREADS_PER_SM    " <<
        deviceProp.maxThreadsPerMultiProcessor << std::endl;



    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    
    /* max threads per block */
    cudaHdrFile << "#define MAX_THREADS_PER_BLOCK    " <<
        deviceProp.maxThreadsPerBlock << std::endl;


    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);

    /* max dim of a thread block */
    cudaHdrFile << "#define MAX_THREADS_DIM_X    " << 
        deviceProp.maxThreadsDim[0] << std::endl;
    cudaHdrFile << "#define MAX_THREADS_DIM_Y    " << 
        deviceProp.maxThreadsDim[1] << std::endl;
    cudaHdrFile << "#define MAX_THREADS_DIM_Z    " << 
        deviceProp.maxThreadsDim[2] << std::endl;



    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);

    /* max dim size of grid */
    cudaHdrFile << "#define MAX_GRIDS_DIM_X    " << 
        deviceProp.maxGridSize[0] << std::endl;
    cudaHdrFile << "#define MAX_GRIDS_DIM_Y    " << 
        deviceProp.maxGridSize[1] << std::endl;
    cudaHdrFile << "#define MAX_GRIDS_DIM_Z    " << 
        deviceProp.maxGridSize[2] << std::endl;


    printf("  Maximum memory pitch:                          %zu bytes\n",
           deviceProp.memPitch);

    /* max mem pitch */
    cudaHdrFile << "#define MEM_PITCH    " << 
        deviceProp.maxThreadsDim[0] << std::endl;


    printf("  Texture alignment:                             %zu bytes\n",
           deviceProp.textureAlignment);

    /* tex alignment */
    cudaHdrFile << "#define TEXTURE_ALIGNMENT    " << 
        deviceProp.textureAlignment << std::endl;



    printf(
        "  Concurrent copy and kernel execution:          %s with %d copy "
        "engine(s)\n",
        (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
    printf("  Run time limit on kernels:                     %s\n",
           deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
    printf("  Integrated GPU sharing Host Memory:            %s\n",
           deviceProp.integrated ? "Yes" : "No");
    printf("  Support host page-locked memory mapping:       %s\n",
           deviceProp.canMapHostMemory ? "Yes" : "No");


    /* map host mem */
    cudaHdrFile << std::endl;
    if (deviceProp.canMapHostMemory)
        cudaHdrFile << "#define PAGE_LOCKED_MEM" << std::endl;


    printf("  Alignment requirement for Surfaces:            %s\n",
           deviceProp.surfaceAlignment ? "Yes" : "No");
    printf("  Device has ECC support:                        %s\n",
           deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
           deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                                : "WDDM (Windows Display Driver Model)");
#endif
    printf("  Device supports Unified Addressing (UVA):      %s\n",
           deviceProp.unifiedAddressing ? "Yes" : "No");

    /* UVA */
    if (deviceProp.unifiedAddressing)
        cudaHdrFile << "#define UVA" << std::endl;


    printf("  Device supports Managed Memory:                %s\n",
           deviceProp.managedMemory ? "Yes" : "No");
    /* Managed mem */
    if (deviceProp.managedMemory)
        cudaHdrFile << "#define MANAGED_MEM" << std::endl;


    printf("  Device supports Compute Preemption:            %s\n",
           deviceProp.computePreemptionSupported ? "Yes" : "No");
    /* computer preemption */
    if (deviceProp.computePreemptionSupported)
        cudaHdrFile << "#define COMPUTE_PREEMPTION" << std::endl;


    printf("  Supports Cooperative Kernel Launch:            %s\n",
           deviceProp.cooperativeLaunch ? "Yes" : "No");

    /* coop kernel */
    if (deviceProp.cooperativeLaunch)
        cudaHdrFile << "#define COOP_KERNEL" << std::endl;


    printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
           deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
    /* coop multi launch */
    if (deviceProp.cooperativeMultiDeviceLaunch) {
        cudaHdrFile << "#define MULTI_DEVICE_COOP_KERNEL" << std::endl;
    }

    printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
           deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

    const char *sComputeMode[] = {
        "Default (multiple host threads can use ::cudaSetDevice() with device "
        "simultaneously)",
        "Exclusive (only one host thread in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this "
        "device)",
        "Exclusive Process (many threads in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Unknown",
        NULL};
    printf("  Compute Mode:\n");
    printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);

  /* Write the header guard #endif */
  cudaHdrFile << std::endl << "#endif" << std::endl;
  cudaHdrFile.close();

  // finish
  exit(EXIT_SUCCESS);
}

