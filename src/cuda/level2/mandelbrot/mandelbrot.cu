////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\mandelbrot\mandelbrot.cu
//
// summary:	Mandelbrot class
// 
//  @file histo-global.cu histogram with global memory atomics
//  
//  origin: (http://selkie.macalester.edu/csinparallel/modules/CUDAArchitecture/build/html/1-Mandelbrot/Mandelbrot.html)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cudacommon.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	BSX
///
/// @brief	block size along
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define BSX 64

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	BSY
///
/// @brief	A macro that defines bsy
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define BSY 4

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	MAX_DEPTH
///
/// @brief	maximum recursion depth
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define MAX_DEPTH 4

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	MIN_SIZE
///
/// @brief	region below which do per-pixel
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define MIN_SIZE 32

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	SUBDIV
///
/// @brief	subdivision factor along each axis
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define SUBDIV 4

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	INIT_SUBDIV
///
/// @brief	subdivision when launched from host
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define INIT_SUBDIV 32
/** binary operation for common dwell "reduction": MAX_DWELL + 1 = neutral
		/// @brief	.
		element, -1 = dwells are different */
#define DIFF_DWELL (-1)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @property	float kernelTime, transferTime
///
/// @brief	Gets the transfer time
///
/// @returns	The transfer time.
////////////////////////////////////////////////////////////////////////////////////////////////////

float kernelTime, transferTime;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @property	cudaEvent_t start, stop
///
/// @brief	Gets the stop
///
/// @returns	The stop.
////////////////////////////////////////////////////////////////////////////////////////////////////

cudaEvent_t start, stop;
/// @brief	The elapsed
float elapsed;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__host__ __device__ int divup(int x, int y)
///
/// @brief	a useful function to compute the number of threads
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	x	The x coordinate. 
/// @param 	y	The y coordinate. 
///
/// @returns	An int.
////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @struct	complex
///
/// @brief	a simple complex type
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

struct complex {

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// @fn	__host__ __device__ complex(float re, float im = 0)
	///
	/// @brief	Constructor
	///
	/// @author	Ed
	/// @date	5/20/2020
	///
	/// @param 	re	The re. 
	/// @param 	im	(Optional) The im. 
	////////////////////////////////////////////////////////////////////////////////////////////////////

	__host__ __device__ complex(float re, float im = 0) {
		this->re = re;
		this->im = im;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// @property	float re, im
	///
	/// @brief	/** real and imaginary part
	///
	/// @returns	The im.
	////////////////////////////////////////////////////////////////////////////////////////////////////

	float re, im;
}; // struct complex

// operator overloads for complex numbers

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	inline __host__ __device__ complex operator+ (const complex &a, const complex &b)
///
/// @brief	Addition operator
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	a	The first value. 
/// @param 	b	A value to add to it. 
///
/// @returns	The result of the operation.
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ complex operator+
(const complex &a, const complex &b) {
	return complex(a.re + b.re, a.im + b.im);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	inline __host__ __device__ complex operator- (const complex &a)
///
/// @brief	Subtraction operator
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	a	A complex to process. 
///
/// @returns	The result of the operation.
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ complex operator-
(const complex &a) { return complex(-a.re, -a.im); }

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	inline __host__ __device__ complex operator- (const complex &a, const complex &b)
///
/// @brief	Subtraction operator
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	a	The first value. 
/// @param 	b	A value to subtract from it. 
///
/// @returns	The result of the operation.
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ complex operator-
(const complex &a, const complex &b) {
	return complex(a.re - b.re, a.im - b.im);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	inline __host__ __device__ complex operator* (const complex &a, const complex &b)
///
/// @brief	Multiplication operator
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	a	The first value to multiply. 
/// @param 	b	The second value to multiply. 
///
/// @returns	The result of the operation.
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ complex operator*
(const complex &a, const complex &b) {
	return complex(a.re * b.re - a.im * b.im, a.im * b.re + a.re * b.im);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	inline __host__ __device__ float abs2(const complex &a)
///
/// @brief	Abs 2
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	a	A complex to process. 
///
/// @returns	A float.
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float abs2(const complex &a) {
	return a.re * a.re + a.im * a.im;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	inline __host__ __device__ complex operator/ (const complex &a, const complex &b)
///
/// @brief	Division operator
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	a	The numerator. 
/// @param 	b	The denominator. 
///
/// @returns	The result of the operation.
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ complex operator/
(const complex &a, const complex &b) {
	float invabs2 = 1 / abs2(b);
	return complex((a.re * b.re + a.im * b.im) * invabs2,
								 (a.im * b.re - b.im * a.re) * invabs2);
}  // operator/

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @def	BS
///
/// @brief	A macro that defines bs
///
/// @author	Ed
/// @date	5/20/2020
////////////////////////////////////////////////////////////////////////////////////////////////////

#define BS 256

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__device__ int pixel_dwell (int w, int h, complex cmin, complex cmax, int x, int y, int MAX_DWELL)
///
/// @brief	computes the dwell for a single pixel
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	w		 	The width. 
/// @param 	h		 	The height. 
/// @param 	cmin	 	The cmin. 
/// @param 	cmax	 	The cmax. 
/// @param 	x		 	The x coordinate. 
/// @param 	y		 	The y coordinate. 
/// @param 	MAX_DWELL	The maximum dwell. 
///
/// @returns	An int.
////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__device__ int same_dwell(int d1, int d2, int MAX_DWELL)
///
/// @brief	Same dwell
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	d1		 	The first int. 
/// @param 	d2		 	The second int. 
/// @param 	MAX_DWELL	The maximum dwell. 
///
/// @returns	An int.
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int same_dwell(int d1, int d2, int MAX_DWELL) {
    int NEUT_DWELL = MAX_DWELL + 1;
	if(d1 == d2)
		return d1;
	else if(d1 == NEUT_DWELL || d2 == NEUT_DWELL)
		return min(d1, d2);
	else
		return DIFF_DWELL;
}  // same_dwell

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__device__ int border_dwell (int w, int h, complex cmin, complex cmax, int x0, int y0, int d, int MAX_DWELL)
///
/// @brief	evaluates the common border dwell, if it exists
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	w		 	The width. 
/// @param 	h		 	The height. 
/// @param 	cmin	 	The cmin. 
/// @param 	cmax	 	The cmax. 
/// @param 	x0		 	The x coordinate 0. 
/// @param 	y0		 	The y coordinate 0. 
/// @param 	d		 	An int to process. 
/// @param 	MAX_DWELL	The maximum dwell. 
///
/// @returns	An int.
////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__global__ void dwell_fill_k (int *dwells, int w, int x0, int y0, int d, int dwell)
///
/// @brief	the kernel to fill the image region with a specific dwell value
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	dwells	If non-null, the dwells. 
/// @param 		   	w	  	The width. 
/// @param 		   	x0	  	The x coordinate 0. 
/// @param 		   	y0	  	The y coordinate 0. 
/// @param 		   	d	  	An int to process. 
/// @param 		   	dwell 	The dwell. 
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void dwell_fill_k
(int *dwells, int w, int x0, int y0, int d, int dwell) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x < d && y < d) {
		x += x0, y += y0;
		dwells[y * w + x] = dwell;
	}
}  // dwell_fill_k

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__global__ void mandelbrot_pixel_k (int *dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d, int MAX_DWELL)
///
/// @brief	/** the kernel to fill in per-pixel values of the portion of the Mandelbrot set
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	dwells   	If non-null, the dwells. 
/// @param 		   	w		 	The width. 
/// @param 		   	h		 	The height. 
/// @param 		   	cmin	 	The cmin. 
/// @param 		   	cmax	 	The cmax. 
/// @param 		   	x0		 	The x coordinate 0. 
/// @param 		   	y0		 	The y coordinate 0. 
/// @param 		   	d		 	An int to process. 
/// @param 		   	MAX_DWELL	The maximum dwell. 
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void mandelbrot_pixel_k
(int *dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d, int MAX_DWELL) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if(x < d && y < d) {
		x += x0, y += y0;
		dwells[y * w + x] = pixel_dwell(w, h, cmin, cmax, x, y, MAX_DWELL);
	}
}  // mandelbrot_pixel_k

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__global__ void mandelbrot_k (int *dwells, int w, int h, complex cmin, complex cmax, int MAX_DWELL)
///
/// @brief	computes the dwells for Mandelbrot image
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	dwells   	the output array. 
/// @param 		   	w		 	the width of the output image. 
/// @param 		   	h		 	the height of the output image. 
/// @param 		   	cmin	 	the complex value associated with the left-bottom corner of the
/// 							image. 
/// @param 		   	cmax	 	the complex value associated with the right-top corner of the
/// 							image. 
/// @param 		   	MAX_DWELL	The maximum dwell. 
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void mandelbrot_k
(int *dwells, int w, int h, complex cmin, complex cmax, int MAX_DWELL) {
	// complex value to start iteration (c)
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int dwell = pixel_dwell(w, h, cmin, cmax, x, y, MAX_DWELL);
	dwells[y * w + x] = dwell;
}  // mandelbrot_k

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	__global__ void mandelbrot_block_k (int *dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d, int depth, int MAX_DWELL)
///
/// @brief	computes the dwells for Mandelbrot image using dynamic parallelism; one block is
/// 		launched per pixel
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	dwells   	the output array. 
/// @param 		   	w		 	the width of the output image. 
/// @param 		   	h		 	the height of the output image. 
/// @param 		   	cmin	 	the complex value associated with the left-bottom corner of the
/// 							image. 
/// @param 		   	cmax	 	the complex value associated with the right-top corner of the
/// 							image. 
/// @param 		   	x0		 	the starting x coordinate of the portion to compute. 
/// @param 		   	y0		 	the starting y coordinate of the portion to compute. 
/// @param 		   	d		 	the size of the portion to compute (the portion is always a
/// 							square) 
/// @param 		   	depth	 	kernel invocation depth. 
/// @param 		   	MAX_DWELL	The maximum dwell. 
///
/// ### remarks	the algorithm reverts to per-pixel Mandelbrot evaluation once either maximum
/// 			depth or minimum size is reached.
////////////////////////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void mandelbrot(int size, int MAX_DWELL)
///
/// @brief	Mandelbrots
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	size	 	The size. 
/// @param 	MAX_DWELL	The maximum dwell. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void mandelbrot(ResultDatabase &resultDB, OptionParser &op, int size, int MAX_DWELL) {
	const bool uvm = op.getOptionBool("uvm");
	const bool uvm_advise = op.getOptionBool("uvm-advise");
	const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
	const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
	int device = 0;
	checkCudaErrors(cudaGetDevice(&device));

	// allocate memory
	int w = size, h = size;
	size_t dwell_sz = w * h * sizeof(int);
	int *h_dwells, *d_dwells;
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
		checkCudaErrors(cudaMallocManaged((void**)&d_dwells, dwell_sz));
	} else {
		checkCudaErrors(cudaMalloc((void**)&d_dwells, dwell_sz));
		h_dwells = (int *)malloc(dwell_sz);
		assert(h_dwells);
	}

	// compute the dwells, copy them back
	dim3 bs(64, 4), grid(divup(w, bs.x), divup(h, bs.y));
    checkCudaErrors(cudaEventRecord(start, 0));
	mandelbrot_k<<<grid, bs>>>
		(d_dwells, w, h, complex(-1.5, -1), complex(0.5, 1), MAX_DWELL);
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
    kernelTime += elapsed * 1.e-3;

    CHECK_CUDA_ERROR();
	checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(start, 0));
	if (uvm) {
		h_dwells = d_dwells;
	} else if (uvm_advise) {
		h_dwells = d_dwells;
		checkCudaErrors(cudaMemAdvise(h_dwells, dwell_sz, cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
		checkCudaErrors(cudaMemAdvise(h_dwells, dwell_sz, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
	} else if (uvm_prefetch) {
		h_dwells = d_dwells;
		checkCudaErrors(cudaMemPrefetchAsync(h_dwells, dwell_sz, device));
	} else if (uvm_prefetch_advise) {
		h_dwells = d_dwells;
		checkCudaErrors(cudaMemAdvise(h_dwells, dwell_sz, cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
		checkCudaErrors(cudaMemAdvise(h_dwells, dwell_sz, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
		checkCudaErrors(cudaMemPrefetchAsync(h_dwells, dwell_sz, device));
	} else {
		checkCudaErrors(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));
	}
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
    transferTime += elapsed * 1.e-3;

	// free data
	checkCudaErrors(cudaFree(d_dwells));
	if (!uvm && !uvm_prefetch && !uvm_advise && !uvm_prefetch_advise) {
		free(h_dwells);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void mandelbrot_dyn(int size, int MAX_DWELL)
///
/// @brief	Mandelbrot dynamic
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param 	size	 	The size. 
/// @param 	MAX_DWELL	The maximum dwell. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void mandelbrot_dyn(ResultDatabase &resultDB, OptionParser &op, int size, int MAX_DWELL) {
	const bool uvm = op.getOptionBool("uvm");
	const bool uvm_advise = op.getOptionBool("uvm-advise");
	const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
	const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
	int device = 0;
	checkCudaErrors(cudaGetDevice(&device));

	// allocate memory
	int w = size, h = size;
	size_t dwell_sz = w * h * sizeof(int);
	int *h_dwells, *d_dwells;
	if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
		checkCudaErrors(cudaMallocManaged((void**)&d_dwells, dwell_sz));
	} else {
		checkCudaErrors(cudaMalloc((void**)&d_dwells, dwell_sz));
		h_dwells = (int *)malloc(dwell_sz);
		assert(h_dwells);
	}

	// compute the dwells, copy them back
	dim3 bs(BSX, BSY), grid(INIT_SUBDIV, INIT_SUBDIV);
    checkCudaErrors(cudaEventRecord(start, 0));
	mandelbrot_block_k<<<grid, bs>>>
		(d_dwells, w, h, complex(-1.5, -1), complex(0.5, 1), 0, 0, w / INIT_SUBDIV, 1, MAX_DWELL);
	checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
    kernelTime += elapsed * 1.e-3;

    CHECK_CUDA_ERROR();
	checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(start, 0));

	if (uvm) {
		h_dwells = d_dwells;
	} else if (uvm_advise) {
		h_dwells = d_dwells;
		checkCudaErrors(cudaMemAdvise(h_dwells, dwell_sz, cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
		checkCudaErrors(cudaMemAdvise(h_dwells, dwell_sz, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
	} else if (uvm_prefetch) {
		h_dwells = d_dwells;
		checkCudaErrors(cudaMemPrefetchAsync(h_dwells, dwell_sz, device));
	} else if (uvm_prefetch_advise) {
		h_dwells = d_dwells;
		checkCudaErrors(cudaMemAdvise(h_dwells, dwell_sz, cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
		checkCudaErrors(cudaMemAdvise(h_dwells, dwell_sz, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
		checkCudaErrors(cudaMemPrefetchAsync(h_dwells, dwell_sz, device));
	} else {
		checkCudaErrors(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));
	}

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
    transferTime += elapsed * 1.e-3;

	// free data
	checkCudaErrors(cudaFree(d_dwells));
	if (!uvm && !uvm_prefetch && !uvm_advise && !uvm_prefetch_advise) {
		free(h_dwells);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void addBenchmarkSpecOptions(OptionParser &op)
///
/// @brief	Adds a benchmark specifier options
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	op	The operation. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("imageSize", OPT_INT, "0", "image height and width");
    op.addOption("iterations", OPT_INT, "0", "iterations of algorithm (the more iterations, the greater speedup from dynamic parallelism)");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// @fn	void RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
///
/// @brief	Executes the benchmark operation
///
/// @author	Ed
/// @date	5/20/2020
///
/// @param [in,out]	resultDB	The result database. 
/// @param [in,out]	op			The operation. 
////////////////////////////////////////////////////////////////////////////////////////////////////

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    printf("Running Mandelbrot\n");

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    bool quiet = op.getOptionBool("quiet");
    int imageSize = op.getOptionInt("imageSize");
    int iters = op.getOptionInt("iterations");
	bool dyn = op.getOptionBool("dyn");
    if (imageSize == 0 || iters == 0) {
        int imageSizes[5] = {2 << 11, 2 << 12, 2 << 13, 2 << 14, 2 << 14};
        int iterSizes[5] = {32, 128, 512, 1024, 8192*16};
        imageSize = imageSizes[op.getOptionInt("size") - 1];
        iters = iterSizes[op.getOptionInt("size") - 1];
    }
    
    if (!quiet) {
        printf("Image Size: %d by %d\n", imageSize, imageSize);
        printf("Num Iterations: %d\n", iters);
		if (dyn) printf("Using dynamic parallelism\n");
        else printf("Not using dynamic parallelism\n");
    }
    
    char atts[1024];
    sprintf(atts, "img:%d,iter:%d", imageSize, iters);

    int passes = op.getOptionInt("passes");
    for (int i = 0; i < passes; i++) {
        if (!quiet) {
            printf("Pass %d:\n", i);
        }

        kernelTime = 0.0f;
        transferTime = 0.0f;
        mandelbrot(resultDB, op, imageSize, iters);
        resultDB.AddResult("mandelbrot_kernel_time", atts, "sec", kernelTime);
        resultDB.AddResult("mandelbrot_transfer_time", atts, "sec", transferTime);
        resultDB.AddResult("mandelbrot_total_time", atts, "sec", transferTime + kernelTime);
        resultDB.AddResult("mandelbrot_parity", atts, "N", transferTime / kernelTime);
        resultDB.AddOverall("Time", "sec", kernelTime+transferTime);
		if (dyn) {
			float totalTime = kernelTime;
			kernelTime = 0.0f;
			transferTime = 0.0f;
			mandelbrot_dyn(resultDB, op, imageSize, iters);
			resultDB.AddResult("mandelbrot_dynpar_kernel_time", atts, "sec", kernelTime);
			resultDB.AddResult("mandelbrot_dynpar_transfer_time", atts, "sec", transferTime);
			resultDB.AddResult("mandelbrot_dynpar_total_time", atts, "sec", transferTime + kernelTime);
			resultDB.AddResult("mandelbrot_dynpar_parity", atts, "N", transferTime / kernelTime);
			resultDB.AddResult("mandelbrot_dynpar_speedup", atts, "N", totalTime/kernelTime);
		}

        if(!quiet) {
            printf("Done.\n");
        }
    }

}
