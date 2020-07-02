////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\common\cudacommon.h
//
// summary:	Declares the cudacommon class
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CUDACOMMON_H
/// <summary>	. </summary>
#define CUDACOMMON_H

// workaround for OS X Snow Leopard w/ gcc 4.2.1 and CUDA 2.3a
// (undefined __sync_fetch_and_add)
#if defined(__APPLE__)
# if _GLIBCXX_ATOMIC_BUILTINS == 1
#undef _GLIBCXX_ATOMIC_BUILTINS
#endif // _GLIBC_ATOMIC_BUILTINS
#endif // __APPLE__

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#if defined __has_include
#  if __has_include ("cuda_device_attr.h")
#    include "cuda_device_attr.h"
#  else
#    error "cuda_device_attr.h can't be found. Please refer to README for instructions."
#  endif
#endif

// On Windows, if we call exit, our console may disappear,
// taking the error message with it, so prompt before exiting.
#if defined(_WIN32)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines safe exit. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="val">	The value. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define safe_exit(val)                          \
{                                               \
    cout << "Press return to exit\n";           \
    cin.get();                                  \
    exit(val);                                  \
}
#else

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines safe exit. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="val">	The value. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define safe_exit(val) exit(val)
#endif

#ifdef UNIFIED_MEMORY
#define ALTIS_CUDA_MALLOC(ptr, size)    checkCudaErrors(cudaMallocManaged((void **)&ptr, (size_t)size))
#define ALTIS_CUDA_FREE(ptr)            checkCudaErrors(cudaFree(ptr))
#else
#define ALTIS_CUDA_MALLOC(ptr, size)    checkCudaErrors(cudaMalloc((void **)&ptr, (size_t)size))
#define ALTIS_CUDA_FREE(ptr)            checkCudaErrors(cudaFree(ptr))
#endif

#ifdef UNIFIED_MEMORY
#define ALTIS_MALLOC(ptr, size)         checkCudaErrors(cudaMallocManaged((void **)&ptr, (size_t)size))
#define ALTIS_FREE(ptr)                 checkCudaErrors(cudaFree(ptr))
#else
#define ALTIS_MALLOC(ptr, size)         ptr = malloc((size_t)size); assert(ptr)
#define ALTIS_FREE(ptr)                 free(ptr)
#endif


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines check cuda error noexit. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA_ERROR_NOEXIT()                                             \
{                                                                             \
    cudaDeviceSynchronize();                                                 \
    cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess)                                                   \
    {                                                                         \
        printf("error=%d name=%s at "                                         \
               "ln: %d\n  ",err,cudaGetErrorString(err),__LINE__);            \
    }                                                                         \
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines check cuda error. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA_ERROR()                                                    \
{                                                                             \
    cudaDeviceSynchronize();                                                 \
    cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess)                                                   \
    {                                                                         \
        printf("error=%d name=%s at "                                         \
               "ln: %d\n  ",err,cudaGetErrorString(err),__LINE__);            \
        safe_exit(EXIT_FAILURE);                                              \
    }                                                                         \
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Another macro that defines check cuda error, no sync. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 6/8/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

    inline void __checkCudaErrors( cudaError err, const char *file, const int line )
    {
        if (cudaSuccess != err) {
	        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                    file, line, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	This will output the proper error string when calling cudaGetLastError. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 6/8/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)
#define _V(x)                      { (x); __getLastDispatchError (#x, __FILE__, __LINE__); }
#define getLastDispatchError(x)    _V(x)

    inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
    {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                    file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }

    inline void __getLastDispatchError( const char *dispatchCall, const char *file, const int line )
    {
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err) {
            fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s execution failed: (%d) %s.\n",
                    file, line, dispatchCall, (int)err, cudaGetErrorString( err ) );
            exit(-1);
        }
    }


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines cuda safe call noexit. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="call">	The call. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define CUDA_SAFE_CALL_NOEXIT(call) do {                                      \
    cudaError err = call;                                                     \
    if (cudaSuccess != err) {                                                 \
        fprintf(stderr, "Cuda error in file '%s' in line %i: %s.\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err) );                \
    }                                                                         \
} while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines cuda safe call. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="call">	The call. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define CUDA_SAFE_CALL(call) do {                                             \
    cudaError err = call;                                                     \
    if (cudaSuccess != err) {                                                 \
        fprintf(stderr, "Cuda error in file '%s' in line %i: %s.\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err) );                \
        safe_exit(EXIT_FAILURE);                                              \
    }                                                                         \
} while (0)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A function that returns the max number of cuda threads per block. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="val">	The max number of cuda threads per block. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

static inline int max_threads_per_block(int device)
{
    cudaDeviceProp prop;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, device));
    return prop.maxThreadsPerBlock;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A function that returns the warp size. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
///
/// <param name="val">	The warp size. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

static inline int warp_size(int device)
{
    cudaDeviceProp prop;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, device));
    return prop.warpSize;
}

// Alleviate aliasing issues

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines restrict. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define RESTRICT __restrict__

#endif // CUDACOMMON_H
