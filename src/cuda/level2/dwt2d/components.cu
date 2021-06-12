/* 
 * Copyright (c) 2009, Jiri Matela
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // file:	altis\src\cuda\level2\dwt2d\dwt_cuda\components.cu
 //
 // summary:	Sort class
 // 
 // origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 
#include <unistd.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>

#include "cudacommon.h"
#include "components.h"
#include "common.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines cuda threads. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define THREADS 256

/* Store 3 RGB float components */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Stores the components in rgb format. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="d_r">	[in,out] If non-null, the r. </param>
/// <param name="d_g">	[in,out] If non-null, the g. </param>
/// <param name="d_b">	[in,out] If non-null, the b. </param>
/// <param name="r">  	A float to process. </param>
/// <param name="g">  	A float to process. </param>
/// <param name="b">  	A float to process. </param>
/// <param name="pos">	The position. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void storeComponents(float *d_r, float *d_g, float *d_b, float r, float g, float b, int pos)
{
    d_r[pos] = (r/255.0f) - 0.5f;
    d_g[pos] = (g/255.0f) - 0.5f;
    d_b[pos] = (b/255.0f) - 0.5f;
}

/* Store 3 RGB intege components */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Stores the components in rgb. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="d_r">	[in,out] If non-null, the r. </param>
/// <param name="d_g">	[in,out] If non-null, the g. </param>
/// <param name="d_b">	[in,out] If non-null, the b. </param>
/// <param name="r">  	An int to process. </param>
/// <param name="g">  	An int to process. </param>
/// <param name="b">  	An int to process. </param>
/// <param name="pos">	The position. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void storeComponents(int *d_r, int *d_g, int *d_b, int r, int g, int b, int pos)
{
    d_r[pos] = r - 128;
    d_g[pos] = g - 128;
    d_b[pos] = b - 128;
} 

/* Store float component */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Stores a component. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="d_c">	[in,out] If non-null, the c. </param>
/// <param name="c">  	A float to process. </param>
/// <param name="pos">	The position. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void storeComponent(float *d_c, float c, int pos)
{
    d_c[pos] = (c/255.0f) - 0.5f;
}

/* Store integer component */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Stores a component. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="d_c">	[in,out] If non-null, the c. </param>
/// <param name="c">  	An int to process. </param>
/// <param name="pos">	The position. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void storeComponent(int *d_c, int c, int pos)
{
    d_c[pos] = c - 128;
}

/* Copy img src data into three separated component buffers */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Copies the source to components. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="d_r">   	[in,out] If non-null, the r. </param>
/// <param name="d_g">   	[in,out] If non-null, the g. </param>
/// <param name="d_b">   	[in,out] If non-null, the b. </param>
/// <param name="d_src"> 	[in,out] If non-null, source for the. </param>
/// <param name="pixels">	The pixels. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void c_CopySrcToComponents(T *d_r, T *d_g, T *d_b, 
                                  unsigned char * d_src, 
                                  int pixels)
{
    int x  = threadIdx.x;
    int gX = blockDim.x*blockIdx.x;

    __shared__ unsigned char sData[THREADS*3];

    /* Copy data to shared mem by 4bytes 
       other checks are not necessary, since 
       d_src buffer is aligned to sharedDataSize */
    if ( (x*4) < THREADS*3 ) {
        float *s = (float *)d_src;
        float *d = (float *)sData;
        d[x] = s[((gX*3)>>2) + x];
    }
    __syncthreads();

    T r, g, b;

    int offset = x*3;
    r = (T)(sData[offset]);
    g = (T)(sData[offset+1]);
    b = (T)(sData[offset+2]);

    int globalOutputPosition = gX + x;
    if (globalOutputPosition < pixels) {
        storeComponents(d_r, d_g, d_b, r, g, b, globalOutputPosition);
    }
}

/* Copy img src data into three separated component buffers */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Copies the source to component. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="d_c">   	[in,out] If non-null, the c. </param>
/// <param name="d_src"> 	[in,out] If non-null, source for the. </param>
/// <param name="pixels">	The pixels. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void c_CopySrcToComponent(T *d_c, unsigned char * d_src, int pixels)
{
    int x  = threadIdx.x;
    int gX = blockDim.x*blockIdx.x;

    __shared__ unsigned char sData[THREADS];

    /* Copy data to shared mem by 4bytes 
       other checks are not necessary, since 
       d_src buffer is aligned to sharedDataSize */
    if ( (x*4) < THREADS) {
        float *s = (float *)d_src;
        float *d = (float *)sData;
        d[x] = s[(gX>>2) + x];
    }
    __syncthreads();

    T c;

    c = (T)(sData[x]);

    int globalOutputPosition = gX + x;
    if (globalOutputPosition < pixels) {
        storeComponent(d_c, c, globalOutputPosition);
    }
}


/* Separate compoents of 8bit RGB source image */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	RGB to components. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="d_r">		   	[in,out] If non-null, the r. </param>
/// <param name="d_g">		   	[in,out] If non-null, the g. </param>
/// <param name="d_b">		   	[in,out] If non-null, the b. </param>
/// <param name="src">		   	[in,out] If non-null, source for the. </param>
/// <param name="width">	   	The width. </param>
/// <param name="height">	   	The height. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void rgbToComponents(T *d_r, T *d_g, T *d_b, unsigned char * src, int width, int height, float &transferTime, float &kernelTime, OptionParser &op)
{
    bool uvm = op.getOptionBool("uvm");
    unsigned char * d_src;
    int pixels      = width*height;
    int alignedSize =  DIVANDRND(width*height, THREADS) * THREADS * 3; //aligned to thread block size -- THREADS

    /* Alloc d_src buffer */
    if (uvm) {
        // do nothing
    } else {
        checkCudaErrors(cudaMalloc((void **)&d_src, alignedSize));
        checkCudaErrors(cudaMemset(d_src, 0, alignedSize));
    }
    /* timing events */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;

    /* Copy data to device */
    cudaEventRecord(start, 0);
    if (uvm) {
        d_src = src;
    } else {
        checkCudaErrors(cudaMemcpy(d_src, src, pixels*3, cudaMemcpyHostToDevice));
    }

    // TODO time needs to be change
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    transferTime += elapsed * 1.e-3;

    /* Kernel */
    dim3 threads(THREADS);
    dim3 grid(alignedSize/(THREADS*3));
    assert(alignedSize%(THREADS*3) == 0);

    cudaEventRecord(start, 0);
    c_CopySrcToComponents<<<grid, threads>>>(d_r, d_g, d_b, d_src, pixels);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();

    /* Free Memory */
    if (uvm) {
        // do nothing
    } else {
        checkCudaErrors(cudaFree(d_src));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	RGB to components. </summary>
///
/// <typeparam name="float">	Type of the float. </typeparam>
/// <param name="d_r">		   	[in,out] If non-null, the r. </param>
/// <param name="d_g">		   	[in,out] If non-null, the g. </param>
/// <param name="d_b">		   	[in,out] If non-null, the b. </param>
/// <param name="src">		   	[in,out] If non-null, source for the. </param>
/// <param name="width">	   	The width. </param>
/// <param name="height">	   	The height. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template void rgbToComponents<float>(float *d_r, float *d_g, float *d_b, unsigned char * src, int width, int height, float &transferTime, float &kernelTime, OptionParser &op);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	RGB to components. </summary>
///
/// <typeparam name="t">	Generic type parameter. </typeparam>
/// <param name="d_r">		   	[in,out] If non-null, the r. </param>
/// <param name="d_g">		   	[in,out] If non-null, the g. </param>
/// <param name="d_b">		   	[in,out] If non-null, the b. </param>
/// <param name="src">		   	[in,out] If non-null, source for the. </param>
/// <param name="width">	   	The width. </param>
/// <param name="height">	   	The height. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template void rgbToComponents<int>(int *d_r, int *d_g, int *d_b, unsigned char * src, int width, int height, float &transferTime, float &kernelTime, OptionParser &op);


/* Copy a 8bit source image data into a color compoment of type T */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Bw to component. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="d_c">		   	[in,out] If non-null, the c. </param>
/// <param name="src">		   	[in,out] If non-null, source for the. </param>
/// <param name="width">	   	The width. </param>
/// <param name="height">	   	The height. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void bwToComponent(T *d_c, unsigned char * src, int width, int height, float &transferTime, float &kernelTime)
{
    unsigned char * d_src;
    int pixels      = width*height;
    int alignedSize =  DIVANDRND(pixels, THREADS) * THREADS; //aligned to thread block size -- THREADS

    /* Alloc d_src buffer */
    checkCudaErrors(cudaMalloc((void **)&d_src, alignedSize));
    checkCudaErrors(cudaMemset(d_src, 0, alignedSize));

    /* timing events */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;

    /* Copy data to device */
    cudaEventRecord(start, 0);
    cudaMemcpy(d_src, src, pixels, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    transferTime += elapsed * 1.e-3;

    /* Kernel */
    dim3 threads(THREADS);
    dim3 grid(alignedSize/(THREADS));
    assert(alignedSize%(THREADS) == 0);

    cudaEventRecord(start, 0);
    c_CopySrcToComponent<<<grid, threads>>>(d_c, d_src, pixels);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();

    /* Free Memory */
    cudaFree(d_src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Bw to component. </summary>
///
/// <typeparam name="float">	Type of the float. </typeparam>
/// <param name="d_c">		   	[in,out] If non-null, the c. </param>
/// <param name="src">		   	[in,out] If non-null, source for the. </param>
/// <param name="width">	   	The width. </param>
/// <param name="height">	   	The height. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template void bwToComponent<float>(float *d_c, unsigned char *src, int width, int height, float &transferTime, float &kernelTime);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Bw to component. </summary>
///
/// <typeparam name="t">	Generic type parameter. </typeparam>
/// <param name="d_c">		   	[in,out] If non-null, the c. </param>
/// <param name="src">		   	[in,out] If non-null, source for the. </param>
/// <param name="width">	   	The width. </param>
/// <param name="height">	   	The height. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template void bwToComponent<int>(int *d_c, unsigned char *src, int width, int height, float &transferTime, float &kernelTime);
