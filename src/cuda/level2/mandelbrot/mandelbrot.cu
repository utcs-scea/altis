/** @file histo-global.cu histogram with global memory atomics */

#include <png.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cudacommon.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

/** block size along */
#define BSX 64
#define BSY 4
/** maximum recursion depth */
#define MAX_DEPTH 4
/** region below which do per-pixel */
#define MIN_SIZE 32
/** subdivision factor along each axis */
#define SUBDIV 4
/** subdivision when launched from host */
#define INIT_SUBDIV 32
/** binary operation for common dwell "reduction": MAX_DWELL + 1 = neutral
		element, -1 = dwells are different */
#define DIFF_DWELL (-1)

float kernelTime, transferTime;
cudaEvent_t start, stop;
float elapsed;

/** a useful function to compute the number of threads */
__host__ __device__ int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

/** a simple complex type */
struct complex {
	__host__ __device__ complex(float re, float im = 0) {
		this->re = re;
		this->im = im;
	}
	/** real and imaginary part */
	float re, im;
}; // struct complex

// operator overloads for complex numbers
inline __host__ __device__ complex operator+
(const complex &a, const complex &b) {
	return complex(a.re + b.re, a.im + b.im);
}
inline __host__ __device__ complex operator-
(const complex &a) { return complex(-a.re, -a.im); }
inline __host__ __device__ complex operator-
(const complex &a, const complex &b) {
	return complex(a.re - b.re, a.im - b.im);
}
inline __host__ __device__ complex operator*
(const complex &a, const complex &b) {
	return complex(a.re * b.re - a.im * b.im, a.im * b.re + a.re * b.im);
}
inline __host__ __device__ float abs2(const complex &a) {
	return a.re * a.re + a.im * a.im;
}
inline __host__ __device__ complex operator/
(const complex &a, const complex &b) {
	float invabs2 = 1 / abs2(b);
	return complex((a.re * b.re + a.im * b.im) * invabs2,
								 (a.im * b.re - b.im * a.re) * invabs2);
}  // operator/
#define BS 256

/** computes the dwell for a single pixel */
__device__ int pixel_dwell
(int w, int h, complex cmin, complex cmax, int x, int y, int MAX_DWELL) {
	complex dc = cmax - cmin;
	float fx = (float)x / w, fy = (float)y / h;
	complex c = cmin + complex(fx * dc.re, fy * dc.im);
	int dwell = 0;
	complex z = c;
	while(dwell < MAX_DWELL && abs2(z) < 2 * 2) {
		z = z * z + c;
		dwell++;
	}
	return dwell;
}  // pixel_dwell

__device__ int same_dwell(int d1, int d2, int MAX_DWELL) {
    int NEUT_DWELL = MAX_DWELL + 1;
	if(d1 == d2)
		return d1;
	else if(d1 == NEUT_DWELL || d2 == NEUT_DWELL)
		return min(d1, d2);
	else
		return DIFF_DWELL;
}  // same_dwell

/** evaluates the common border dwell, if it exists */
__device__ int border_dwell
(int w, int h, complex cmin, complex cmax, int x0, int y0, int d, int MAX_DWELL) {
	// check whether all boundary pixels have the same dwell
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int bs = blockDim.x * blockDim.y;
	int comm_dwell = MAX_DWELL + 1;
	// for all boundary pixels, distributed across threads
	for(int r = tid; r < d; r += bs) {
		// for each boundary: b = 0 is east, then counter-clockwise
		for(int b = 0; b < 4; b++) {
			int x = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
			int y = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
			int dwell = pixel_dwell(w, h, cmin, cmax, x, y, MAX_DWELL);
			comm_dwell = same_dwell(comm_dwell, dwell, MAX_DWELL);
		}
	}  // for all boundary pixels
	// reduce across threads in the block
	__shared__ int ldwells[BSX * BSY];
	int nt = min(d, BSX * BSY);
	if(tid < nt)
		ldwells[tid] = comm_dwell;
	__syncthreads();
	for(; nt > 1; nt /= 2) {
		if(tid < nt / 2)
			ldwells[tid] = same_dwell(ldwells[tid], ldwells[tid + nt / 2], MAX_DWELL);
		__syncthreads();
	}
	return ldwells[0];
}  // border_dwell

/** the kernel to fill the image region with a specific dwell value */
__global__ void dwell_fill_k
(int *dwells, int w, int x0, int y0, int d, int dwell) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x < d && y < d) {
		x += x0, y += y0;
		dwells[y * w + x] = dwell;
	}
}  // dwell_fill_k

/** the kernel to fill in per-pixel values of the portion of the Mandelbrot set
		*/
__global__ void mandelbrot_pixel_k
(int *dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d, int MAX_DWELL) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if(x < d && y < d) {
		x += x0, y += y0;
		dwells[y * w + x] = pixel_dwell(w, h, cmin, cmax, x, y, MAX_DWELL);
	}
}  // mandelbrot_pixel_k

/** computes the dwells for Mandelbrot image 
		@param dwells the output array
		@param w the width of the output image
		@param h the height of the output image
		@param cmin the complex value associated with the left-bottom corner of the
		image
		@param cmax the complex value associated with the right-top corner of the
		image
 */
__global__ void mandelbrot_k
(int *dwells, int w, int h, complex cmin, complex cmax, int MAX_DWELL) {
	// complex value to start iteration (c)
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int dwell = pixel_dwell(w, h, cmin, cmax, x, y, MAX_DWELL);
	dwells[y * w + x] = dwell;
}  // mandelbrot_k

/** computes the dwells for Mandelbrot image using dynamic parallelism; one
		block is launched per pixel
		@param dwells the output array
		@param w the width of the output image
		@param h the height of the output image
		@param cmin the complex value associated with the left-bottom corner of the
		image
		@param cmax the complex value associated with the right-top corner of the
		image
		@param x0 the starting x coordinate of the portion to compute
		@param y0 the starting y coordinate of the portion to compute
		@param d the size of the portion to compute (the portion is always a square)
		@param depth kernel invocation depth
		@remarks the algorithm reverts to per-pixel Mandelbrot evaluation once
		either maximum depth or minimum size is reached
 */
__global__ void mandelbrot_block_k
(int *dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, 
 int d, int depth, int MAX_DWELL) {
	x0 += d * blockIdx.x, y0 += d * blockIdx.y;
	int comm_dwell = border_dwell(w, h, cmin, cmax, x0, y0, d, MAX_DWELL);
	if(threadIdx.x == 0 && threadIdx.y == 0) {
		if(comm_dwell != DIFF_DWELL) {
			// uniform dwell, just fill
			dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
			dwell_fill_k<<<grid, bs>>>(dwells, w, x0, y0, d, comm_dwell);
		} else if(depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
			// subdivide recursively
			dim3 bs(blockDim.x, blockDim.y), grid(SUBDIV, SUBDIV);
			mandelbrot_block_k<<<grid, bs>>>
				(dwells, w, h, cmin, cmax, x0, y0, d / SUBDIV, depth + 1, MAX_DWELL);
		} else {
			// leaf, per-pixel kernel
			dim3 bs(BSX, BSY), grid(divup(d, BSX), divup(d, BSY));
			mandelbrot_pixel_k<<<grid, bs>>>
				(dwells, w, h, cmin, cmax, x0, y0, d, MAX_DWELL);
		}
		//CUDA_SAFE_CALL(cudaGetLastError());
	}
}  // mandelbrot_block_k

void mandelbrot(int size, int MAX_DWELL) {
	// allocate memory
	int w = size, h = size;
	size_t dwell_sz = w * h * sizeof(int);
	int *h_dwells, *d_dwells;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_dwells, dwell_sz));
	h_dwells = (int*)malloc(dwell_sz);

	// compute the dwells, copy them back
	dim3 bs(64, 4), grid(divup(w, bs.x), divup(h, bs.y));
    cudaEventRecord(start, 0);
	mandelbrot_k<<<grid, bs>>>
		(d_dwells, w, h, complex(-1.5, -1), complex(0.5, 1), MAX_DWELL);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernelTime += elapsed * 1.e-3;

    CHECK_CUDA_ERROR();
	CUDA_SAFE_CALL(cudaThreadSynchronize());
    cudaEventRecord(start, 0);
	CUDA_SAFE_CALL(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    transferTime += elapsed * 1.e-3;

	// free data
	cudaFree(d_dwells);
	free(h_dwells);
}

void mandelbrot_dyn(int size, int MAX_DWELL) {
	// allocate memory
	int w = size, h = size;
	size_t dwell_sz = w * h * sizeof(int);
	int *h_dwells, *d_dwells;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_dwells, dwell_sz));
	h_dwells = (int*)malloc(dwell_sz);

	// compute the dwells, copy them back
	dim3 bs(BSX, BSY), grid(INIT_SUBDIV, INIT_SUBDIV);
    cudaEventRecord(start, 0);
	mandelbrot_block_k<<<grid, bs>>>
		(d_dwells, w, h, complex(-1.5, -1), complex(0.5, 1), 0, 0, w / INIT_SUBDIV, 1, MAX_DWELL);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernelTime += elapsed * 1.e-3;

    CHECK_CUDA_ERROR();
	CUDA_SAFE_CALL(cudaThreadSynchronize());
    cudaEventRecord(start, 0);
	CUDA_SAFE_CALL(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    transferTime += elapsed * 1.e-3;

	// free data
	cudaFree(d_dwells);
	free(h_dwells);
}

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("imageSize", OPT_INT, "0", "image height and width");
    op.addOption("iterations", OPT_INT, "0", "iterations of algorithm (the more iterations, the greater speedup from dynamic parallelism)");
}

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    printf("Running Mandelbrot\n");

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    bool quiet = op.getOptionBool("quiet");
    int imageSize = op.getOptionInt("imageSize");
    int iters = op.getOptionInt("iterations");
    if(imageSize == 0 || iters == 0) {
        int imageSizes[4] = {2 << 11, 2 << 12, 2 << 13, 2 << 14};
        int iterSizes[4] = {32, 128, 512, 1024};
        imageSize = imageSizes[op.getOptionInt("size") - 1];
        iters = iterSizes[op.getOptionInt("size") - 1];
    }
    
    if(!quiet) {
        printf("Image Size: %d by %d\n", imageSize, imageSize);
        printf("Num Iterations: %d\n", iters);
#ifdef DYNAMIC_PARALLELISM
        printf("Using dynamic parallelism\n");
#else
        printf("Not using dynamic parallelism\n");
#endif
    }
    
    char atts[1024];
    sprintf(atts, "img:%d,iter:%d", imageSize, iters);

    int passes = op.getOptionInt("passes");
    for(int i = 0; i < passes; i++) {
        if(!quiet) {
            printf("Pass %d:\n", i);
        }

        kernelTime = 0.0f;
        transferTime = 0.0f;
        mandelbrot(imageSize, iters);
        resultDB.AddResult("mandelbrot_kernel_time", atts, "sec", kernelTime);
        resultDB.AddResult("mandelbrot_transfer_time", atts, "sec", transferTime);
        resultDB.AddResult("mandelbrot_total_time", atts, "sec", transferTime + kernelTime);
        resultDB.AddResult("mandelbrot_parity", atts, "N", transferTime / kernelTime);
        resultDB.AddOverall("Time", "sec", kernelTime+transferTime);
        /*
#ifdef DYNAMIC_PARALLELISM
        float totalTime = kernelTime;
        kernelTime = 0.0f;
        transferTime = 0.0f;
        mandelbrot_dyn(imageSize, iters);
        resultDB.AddResult("mandelbrot_dynpar_kernel_time", atts, "sec", kernelTime);
        resultDB.AddResult("mandelbrot_dynpar_transfer_time", atts, "sec", transferTime);
        resultDB.AddResult("mandelbrot_dynpar_total_time", atts, "sec", transferTime + kernelTime);
        resultDB.AddResult("mandelbrot_dynpar_parity", atts, "N", transferTime / kernelTime);
        resultDB.AddResult("mandelbrot_dynpar_speedup", atts, "N", totalTime/kernelTime);
#endif
*/

        if(!quiet) {
            printf("Done.\n");
        }
    }

}
