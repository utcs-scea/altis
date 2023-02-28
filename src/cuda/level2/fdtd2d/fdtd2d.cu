/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 *
 * Modfified by Bodun Hu <bodunhu@utexas.edu>
 * Added: UVM and coop support
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include "polybenchUtilFuncts.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"

#include <cooperative_groups.h>
using namespace cooperative_groups;

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float, int, and double */
typedef float DATA_TYPE;

struct fdtd_params {
    int NX;
    int NY;
    DATA_TYPE *_fict_;
    DATA_TYPE *ex;
    DATA_TYPE *ey;
    DATA_TYPE *hz;
    int t;
};


void init_arrays(size_t NX, size_t NY, size_t tmax, DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
    int i, j;

    for (i = 0; i < tmax; i++)
    {
        _fict_[i] = (DATA_TYPE) i;
    }
    
    for (i = 0; i < NX; i++)
    {
        for (j = 0; j < NY; j++)
        {
            ex[i*NY + j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
            ey[i*NY + j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
            hz[i*NY + j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
        }
    }
}


void runFdtd(size_t NX, size_t NY, size_t tmax, DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
    int t, i, j;
    
    for (t=0; t < tmax; t++)  
    {
        for (j=0; j < NY; j++)
        {
            ey[0*NY + j] = _fict_[t];
        }
    
        for (i = 1; i < NX; i++)
        {
            for (j = 0; j < NY; j++)
            {
                ey[i*NY + j] = ey[i*NY + j] - 0.5*(hz[i*NY + j] - hz[(i-1)*NY + j]);
            }
        }

        for (i = 0; i < NX; i++)
        {
            for (j = 1; j < NY; j++)
            {
                ex[i*(NY+1) + j] = ex[i*(NY+1) + j] - 0.5*(hz[i*NY + j] - hz[i*NY + (j-1)]);
            }
        }

        for (i = 0; i < NX; i++)
        {
            for (j = 0; j < NY; j++)
            {
                hz[i*NY + j] = hz[i*NY + j] - 0.7*(ex[i*(NY+1) + (j+1)] - ex[i*(NY+1) + j] + ey[(i+1)*NY + j] - ey[i*NY + j]);
            }
        }
    }
}


void compareResults(size_t NX, size_t NY, DATA_TYPE* hz1, DATA_TYPE* hz2)
{
    int i, j, fail;
    fail = 0;
    
    for (i=0; i < NX; i++) 
    {
        for (j=0; j < NY; j++) 
        {
            if (percentDiff(hz1[i*NY + j], hz2[i*NY + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
            {
                fail++;
            }
        }
    }
    
    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

__global__ void fdtd_step1_kernel(size_t NX, size_t NY, DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < NX) && (j < NY))
    {
        if (i == 0) 
        {
            ey[i * NY + j] = _fict_[t];
        }
        else
        { 
            ey[i * NY + j] = ey[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[(i-1) * NY + j]);
        }
    }
}



__global__ void fdtd_step2_kernel(size_t NX, size_t NY, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if ((i < NX) && (j < NY) && (j > 0))
    {
        ex[i * (NY+1) + j] = ex[i * (NY+1) + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
    }
}


__global__ void fdtd_step3_kernel(size_t NX, size_t NY, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if ((i < NX) && (j < NY))
    {	
        hz[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * (NY+1) + (j+1)] - ex[i * (NY+1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
    }
}

__global__ void fdtd_coop_kernel(fdtd_params params)
{
    int NX = params.NX;
    int NY = params.NY;
    DATA_TYPE *_fict_ = params._fict_;
    DATA_TYPE *ex = params.ex;
    DATA_TYPE *ey = params.ey;
    DATA_TYPE *hz = params.hz;
    int t = params.t;
    
    grid_group g = this_grid();
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // kernel 1
    if ((i < NX) && (j < NY))
    {
        if (i == 0) 
        {
            ey[i * NY + j] = _fict_[t];
        }
        else
        { 
            ey[i * NY + j] = ey[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[(i-1) * NY + j]);
        }
    }

    // kernel 2
    if ((i < NX) && (j < NY) && (j > 0))
    {
        ex[i * (NY+1) + j] = ex[i * (NY+1) + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
    }

    g.sync();

    // kernel 3
    if ((i < NX) && (j < NY))
    {
        hz[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * (NY+1) + (j+1)] - ex[i * (NY+1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
    }
}


void fdtdCuda(size_t NX, size_t NY, size_t tmax, DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, DATA_TYPE* hz_outputFromGpu,
            ResultDatabase &DB, OptionParser &op)
{
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    int device = 0;
    checkCudaErrors(cudaGetDevice(&device));

    double t_start, t_end;

    DATA_TYPE *_fict_gpu;
    DATA_TYPE *ex_gpu;
    DATA_TYPE *ey_gpu;
    DATA_TYPE *hz_gpu;


    checkCudaErrors(cudaMalloc((void **)&_fict_gpu, sizeof(DATA_TYPE) * tmax));
    checkCudaErrors(cudaMalloc((void **)&ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1)));
    checkCudaErrors(cudaMalloc((void **)&ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY));
    checkCudaErrors(cudaMalloc((void **)&hz_gpu, sizeof(DATA_TYPE) * NX * NY));

    checkCudaErrors(cudaMemcpy(_fict_gpu, _fict_, sizeof(DATA_TYPE) * tmax, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ex_gpu, ex, sizeof(DATA_TYPE) * NX * (NY + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ey_gpu, ey, sizeof(DATA_TYPE) * (NX + 1) * NY, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice));

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid( (size_t)ceil(((float)NY) / ((float)block.x)), (size_t)ceil(((float)NX) / ((float)block.y)));

    t_start = rtclock();

    if (op.getOptionBool("coop"))
    {
        fdtd_params params;
        params.NX = NX;
        params.NY = NY;
        params._fict_ = _fict_gpu;
        params.ex = ex_gpu;
        params.ey = ey_gpu;
        params.hz = hz_gpu;
        void *p_params = {&params};
        for (int t = 0; t < tmax; t++)
        {
            params.t = t;
            checkCudaErrors(cudaLaunchCooperativeKernel((void*)fdtd_coop_kernel, grid, block, &p_params));
        }
    }
    else
    {
        cudaStream_t stream1, stream2;
        checkCudaErrors(cudaStreamCreate(&stream1));
        checkCudaErrors(cudaStreamCreate(&stream2));
        for (int t = 0; t < tmax; t++)
        {
            fdtd_step1_kernel<<<grid,block,0,stream1>>>(NX, NY, _fict_gpu, ex_gpu, ey_gpu, hz_gpu, t);
            fdtd_step2_kernel<<<grid,block,0,stream2>>>(NX, NY, ex_gpu, ey_gpu, hz_gpu, t);
            fdtd_step3_kernel<<<grid,block>>>(NX, NY, ex_gpu, ey_gpu, hz_gpu, t);
        }
        checkCudaErrors(cudaStreamDestroy(stream1));
        checkCudaErrors(cudaStreamDestroy(stream2));
    }
    cudaDeviceSynchronize();
    t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

    checkCudaErrors(cudaMemcpy(hz_outputFromGpu, hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(_fict_gpu));
    checkCudaErrors(cudaFree(ex_gpu));
    checkCudaErrors(cudaFree(ey_gpu));
    checkCudaErrors(cudaFree(hz_gpu));
}

void fdtdCudaUnifiedMemory(size_t NX, size_t NY, size_t tmax, DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz,
    ResultDatabase &DB, OptionParser &op)
{
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    int device = 0;
    checkCudaErrors(cudaGetDevice(&device));

    double t_start, t_end;

    DATA_TYPE *_fict_gpu;
    DATA_TYPE *ex_gpu;
    DATA_TYPE *ey_gpu;
    DATA_TYPE *hz_gpu;

    _fict_gpu = _fict_;
    ex_gpu = ex;
    ey_gpu = ey;
    hz_gpu = hz;

    if (uvm)
    {
        // Do nothing
    }
    else if (uvm_advise)
    {
        checkCudaErrors(cudaMemAdvise(_fict_gpu, sizeof(DATA_TYPE) * tmax, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(_fict_gpu, sizeof(DATA_TYPE) * tmax, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemAdvise(_fict_gpu, sizeof(DATA_TYPE) * tmax, cudaMemAdviseSetAccessedBy, device));

        checkCudaErrors(cudaMemAdvise(ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1), cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemAdvise(ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1), cudaMemAdviseSetAccessedBy, device));

        checkCudaErrors(cudaMemAdvise(ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemAdvise(ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY, cudaMemAdviseSetAccessedBy, device));

        checkCudaErrors(cudaMemAdvise(hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemAdvise(hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemAdviseSetAccessedBy, device));
    }
    else if (uvm_prefetch)
    {
        checkCudaErrors(cudaMemPrefetchAsync(_fict_gpu, sizeof(DATA_TYPE) * tmax, device));
        checkCudaErrors(cudaMemPrefetchAsync(ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1), device));
        checkCudaErrors(cudaMemPrefetchAsync(ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY, device));
        checkCudaErrors(cudaMemPrefetchAsync(hz_gpu,  sizeof(DATA_TYPE) * NX * NY, device));
    }
    else if (uvm_prefetch_advise)
    {
        checkCudaErrors(cudaMemAdvise(_fict_gpu, sizeof(DATA_TYPE) * tmax, cudaMemAdviseSetReadMostly, device));
        checkCudaErrors(cudaMemAdvise(_fict_gpu, sizeof(DATA_TYPE) * tmax, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemAdvise(_fict_gpu, sizeof(DATA_TYPE) * tmax, cudaMemAdviseSetAccessedBy, device));

        checkCudaErrors(cudaMemAdvise(ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1), cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemAdvise(ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1), cudaMemAdviseSetAccessedBy, device));

        checkCudaErrors(cudaMemAdvise(ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemAdvise(ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY, cudaMemAdviseSetAccessedBy, device));

        checkCudaErrors(cudaMemAdvise(hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemAdviseSetPreferredLocation, device));
        checkCudaErrors(cudaMemAdvise(hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemAdviseSetAccessedBy, device));

        checkCudaErrors(cudaMemPrefetchAsync(_fict_gpu, sizeof(DATA_TYPE) * tmax, device));
        checkCudaErrors(cudaMemPrefetchAsync(ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1), device));
        checkCudaErrors(cudaMemPrefetchAsync(ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY, device));
        checkCudaErrors(cudaMemPrefetchAsync(hz_gpu,  sizeof(DATA_TYPE) * NX * NY, device));
    }
    else
    {
        std::cerr << "unrecognized uvm flag, exiting..." << std::endl;
        exit(-1);
    }

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid( (size_t)ceil(((float)NY) / ((float)block.x)), (size_t)ceil(((float)NX) / ((float)block.y)));

    // cudaStream_t stream1, stream2;
    // checkCudaErrors(cudaStreamCreate(&stream1));
    // checkCudaErrors(cudaStreamCreate(&stream2));
    t_start = rtclock();

    if (op.getOptionBool("coop"))
    {
        fdtd_params params;
        params.NX = NX;
        params.NY = NY;
        params._fict_ = _fict_gpu;
        params.ex = ex_gpu;
        params.ey = ey_gpu;
        params.hz = hz_gpu;
        void *p_params = {&params};
        for (int t = 0; t < tmax; t++)
        {
            params.t = t;
            checkCudaErrors(cudaLaunchCooperativeKernel((void*)fdtd_coop_kernel, grid, block, &p_params));
        }
    }
    else
    {
        cudaStream_t stream1, stream2;
        checkCudaErrors(cudaStreamCreate(&stream1));
        checkCudaErrors(cudaStreamCreate(&stream2));
        for (int t = 0; t < tmax; t++)
        {
            fdtd_step1_kernel<<<grid,block,0,stream1>>>(NX, NY, _fict_gpu, ex_gpu, ey_gpu, hz_gpu, t);
            fdtd_step2_kernel<<<grid,block,0,stream1>>>(NX, NY, ex_gpu, ey_gpu, hz_gpu, t);
            fdtd_step3_kernel<<<grid,block>>>(NX, NY, ex_gpu, ey_gpu, hz_gpu, t);
        }
        checkCudaErrors(cudaStreamDestroy(stream1));
        checkCudaErrors(cudaStreamDestroy(stream2));
    }

    cudaDeviceSynchronize();
    t_end = rtclock();
    fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
    {
    // Do nothing
    }

    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
    {
        checkCudaErrors(cudaMemAdvise(hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
        checkCudaErrors(cudaMemAdvise(hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        checkCudaErrors(cudaMemAdvise(hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
        checkCudaErrors(cudaMemPrefetchAsync(hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaCpuDeviceId));
        checkCudaErrors(cudaThreadSynchronize());
    }
}

void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("uvm", OPT_BOOL, "0", "enable CUDA Unified Virtual Memory, only demand paging");
    op.addOption("uvm-advise", OPT_BOOL, "0", "guide the driver about memory usage patterns");
    op.addOption("uvm-prefetch", OPT_BOOL, "0", "prefetch memory the specified destination device");
    op.addOption("uvm-prefetch-advise", OPT_BOOL, "0", "prefetch memory the specified destination device with memory guidance on");
    op.addOption("coop", OPT_BOOL, "0", "use cooperative kernel instead normal kernels");
    op.addOption("compare", OPT_BOOL, "0", "compare GPU output with CPU output");
}

void RunBenchmark(ResultDatabase &DB, OptionParser &op)
{
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    const bool compare = op.getOptionBool("compare");

    const size_t s = 5;
    size_t NX_sizes[s] = {100, 1000, 2000, 8000, 16000};
    size_t NY_sizes[s] = {200, 1200, 2600, 9600, 20000};
    size_t tmax_sizes[s] =  {240, 500, 1000, 4000, 8000};

    size_t NX = NX_sizes[op.getOptionInt("size") - 1];
    size_t NY = NY_sizes[op.getOptionInt("size") - 1];
    size_t tmax = tmax_sizes[op.getOptionInt("size") - 1];

    double t_start, t_end;

    DATA_TYPE* _fict_;
    DATA_TYPE* ex;
    DATA_TYPE* ey;
    DATA_TYPE* hz;
    DATA_TYPE* hz_outputFromGpu;

    if (compare)
    {
        if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
        {
            DATA_TYPE* _fict_gpu;
            DATA_TYPE* ex_gpu;
            DATA_TYPE* ey_gpu;
            DATA_TYPE* hz_gpu;
            checkCudaErrors(cudaMallocManaged(&_fict_gpu, tmax*sizeof(DATA_TYPE)));
            checkCudaErrors(cudaMallocManaged(&ex_gpu, NX*(NY+1)*sizeof(DATA_TYPE)));
            checkCudaErrors(cudaMallocManaged(&ey_gpu, (NX+1)*NY*sizeof(DATA_TYPE)));
            checkCudaErrors(cudaMallocManaged(&hz_gpu, NX*NY*sizeof(DATA_TYPE)));

            _fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
            assert(_fict_);
            ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
            assert(ex);
            ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
            assert(ey);
            hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
            assert(hz);

            init_arrays(NX, NY, tmax, _fict_gpu, ex_gpu, ey_gpu, hz_gpu);
            checkCudaErrors(cudaMemcpy(_fict_, _fict_gpu, tmax*sizeof(DATA_TYPE), cudaMemcpyHostToHost));
            checkCudaErrors(cudaMemcpy(ex, ex_gpu, NX*(NY+1)*sizeof(DATA_TYPE), cudaMemcpyHostToHost));
            checkCudaErrors(cudaMemcpy(ey, ey_gpu, (NX+1)*NY*sizeof(DATA_TYPE), cudaMemcpyHostToHost));
            checkCudaErrors(cudaMemcpy(hz, hz_gpu, NX*NY*sizeof(DATA_TYPE), cudaMemcpyHostToHost));
            
            fdtdCudaUnifiedMemory(NX, NY, tmax, _fict_gpu, ex_gpu, ey_gpu, hz_gpu, DB, op);
            t_start = rtclock();
            runFdtd(NX, NY, tmax, _fict_, ex, ey, hz);
            t_end = rtclock();
            fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
            compareResults(NX, NY, hz, hz_gpu);

            checkCudaErrors(cudaFree(_fict_gpu));
            checkCudaErrors(cudaFree(ex_gpu));
            checkCudaErrors(cudaFree(ey_gpu));
            checkCudaErrors(cudaFree(hz_gpu));
            free(_fict_);
            free(ex);
            free(ey);
            free(hz);
        }
        else
        {
            _fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
            assert(_fict_);
            ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
            assert(ex);
            ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
            assert(ey);
            hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
            assert(hz);
            hz_outputFromGpu = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
            assert(hz_outputFromGpu);

            init_arrays(NX, NY, tmax, _fict_, ex, ey, hz);
            fdtdCuda(NX, NY, tmax, _fict_, ex, ey, hz, hz_outputFromGpu, DB, op);
            t_start = rtclock();
            runFdtd(NX, NY, tmax, _fict_, ex, ey, hz);
            t_end = rtclock();
            fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
            compareResults(NX, NY, hz, hz_outputFromGpu);

            free(_fict_);
            free(ex);
            free(ey);
            free(hz);
            free(hz_outputFromGpu);
        }
    }
    else
    {
        if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
        {
            checkCudaErrors(cudaMallocManaged(&_fict_, tmax*sizeof(DATA_TYPE)));
            checkCudaErrors(cudaMallocManaged(&ex, NX*(NY+1)*sizeof(DATA_TYPE)));
            checkCudaErrors(cudaMallocManaged(&ey, (NX+1)*NY*sizeof(DATA_TYPE)));
            checkCudaErrors(cudaMallocManaged(&hz, NX*NY*sizeof(DATA_TYPE)));

            init_arrays(NX, NY, tmax, _fict_, ex, ey, hz);
            fdtdCudaUnifiedMemory(NX, NY, tmax, _fict_, ex, ey, hz, DB, op);

            checkCudaErrors(cudaFree(_fict_));
            checkCudaErrors(cudaFree(ex));
            checkCudaErrors(cudaFree(ey));
            checkCudaErrors(cudaFree(hz));
        }
        else
        {
            _fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
            assert(_fict_);
            ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
            assert(ex);
            ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
            assert(ey);
            hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
            assert(hz);
            hz_outputFromGpu = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
            assert(hz_outputFromGpu);

            init_arrays(NX, NY, tmax, _fict_, ex, ey, hz);
            fdtdCuda(NX, NY, tmax, _fict_, ex, ey, hz, hz_outputFromGpu, DB, op);

            free(_fict_);
            free(ex);
            free(ey);
            free(hz);
            free(hz_outputFromGpu);
        }
    }
}
