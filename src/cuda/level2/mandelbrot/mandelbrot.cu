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

float kernelTime, transferTime;
cudaEvent_t start, stop;
float elapsed;

/** a useful function to compute the number of threads */
int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

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

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("imageSize", OPT_INT, "0", "image height and width");
    op.addOption("iterations", OPT_INT, "0", "iterations of algorithm (the more iterations, the greater speedup from dynamic parallelism)");
}

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int imageSize = op.getOptionInt("imageSize");
    int iters = op.getOptionInt("iterations");
    if(imageSize == 0 || iters == 0) {
        int imageSizes[4] = {2 << 11, 2 << 13, 2 << 14, 2 << 15};
        int iterSizes[4] = {32, 128, 512, 1024};
        imageSize = imageSizes[op.getOptionInt("size") - 1];
        iters = iterSizes[op.getOptionInt("size") - 1];
    }
    
    printf("Image Size: %d by %d\n", imageSize, imageSize);
    printf("Num Iterations: %d\n", iters);

    int passes = op.getOptionInt("passes");
    for(int i = 0; i < passes; i++) {
        kernelTime = 0.0f;
        transferTime = 0.0f;
        printf("Pass %d:\n", i);
        mandelbrot(imageSize, iters);
        printf("Done.\n");
        char atts[1024];
        sprintf(atts, "img:%d,iter:%d", imageSize, iters);
        resultDB.AddResult("mandelbrot_kernel_time", atts, "sec", kernelTime);
        resultDB.AddResult("mandelbrot_transfer_time", atts, "sec", transferTime);
        resultDB.AddResult("mandelbrot_parity", atts, "N", transferTime / kernelTime);
    }

}
