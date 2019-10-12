
/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <iostream>
#include <cstdlib>
#include <string>
#include <math.h>
#include <cassert>
#include <cuda.h>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include "fdtd2d.h"
#include<cuda_profiler_api.h>

#define DEFAULT_GPU 2

// TODO subject to change
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

#define SMALL_FLOAT_VAL 0.00000001f
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define tmax 2//500
#define NX 2048
#define NY 2048


float absVal(float a);
float percentDiff(double val1, double val2);

using namespace std;


float absVal(float a) {
    if (a < 0) return (-1*a);
    else       return a;
}

float percentDiff(double val1, double val2) {
    if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01))
        return 0.0f;
    else
        return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
} 


void addBenchmarkSpecOptions(OptionParser &op) {
   // TODO, maybe add benchmark specs 
}

void init_arrays(DATA_TYPE *_fict_, DATA_TYPE *ex,
        DATA_TYPE *ey, DATA_TYPE *hz) {
    assert(_fict_ && ex && ey && hz);

    int i = 0;
    for (; i < tmax; i++) {
        _fict_[i] = (DATA_TYPE)i;
    }

    int j;
    for (i = 0; i < NX; i++) {
        for (j = 0; j < NY; j++) {
            int index_to_update = i * NY + j;
            ex[index_to_update] = ((DATA_TYPE)i*(j+1)+1) / NX;
            ey[index_to_update] = ((DATA_TYPE)(i-1)*(j+2)+2) / NX;
            hz[index_to_update] = ((DATA_TYPE)(i-9)*(j+4)+3) / NX;
        }
    }
}

void run_fdtd_cpu(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
{
    assert(_fict_ && ex && ey && hz);
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


void compareResults(DATA_TYPE* hz1, DATA_TYPE* hz2)
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
	cout << "Non-Matching CPU-GPU Outputs Beyond Error Threshold of " <<
        PERCENT_DIFF_ERROR_THRESHOLD << " Percent: " << fail << endl;
}

__global__ void kernel1(DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
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

__global__ void kernel2(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NX) && (j < NY) && (j > 0))
	{
		ex[i * (NY+1) + j] = ex[i * (NY+1) + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
	}
}

__global__ void kernel3(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NX) && (j < NY))
	{
		hz[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * (NY+1) + (j+1)] - ex[i * (NY+1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
	}
}

void run_fdtd_cuda(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, DATA_TYPE *hz_from_gpu) {
    assert(_fict_ && ex && ey && hz && hz_from_gpu);

    DATA_TYPE *_fict_gpu = NULL;
    DATA_TYPE *ex_gpu = NULL;
    DATA_TYPE *ey_gpu = NULL;
    DATA_TYPE *hz_gpu = NULL;

    // allocating resources
    cudaMalloc((void **)&_fict_gpu, sizeof(DATA_TYPE) * tmax);
    if (cudaGetLastError() != cudaSuccess)
    {
        cout << "cudaMalloc d_A returned error code" << endl;
        exit(1);
    }

	cudaMalloc((void **)&ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1));
    if (cudaGetLastError() != cudaSuccess)
    {
        cout << "cudaMalloc d_A returned error code" << endl;
        exit(1);
    }


	cudaMalloc((void **)&ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY);
    if (cudaGetLastError() != cudaSuccess)
    {
        cout << "cudaMalloc d_A returned error code" << endl;
        exit(1);
    }

	cudaMalloc((void **)&hz_gpu, sizeof(DATA_TYPE) * NX * NY);
    if (cudaGetLastError() != cudaSuccess)
    {
        cout << "cudaMalloc d_A returned error code" << endl;
        exit(1);
    }

    // copy data to device
	cudaMemcpy(_fict_gpu, _fict_, sizeof(DATA_TYPE) * tmax, cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
    {
        cout << "cudaMemcpy d_A returned error code" << endl;
        exit(1);
    }
	
    cudaMemcpy(ex_gpu, ex, sizeof(DATA_TYPE) * NX * (NY + 1), cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
    {
        cout << "cudaMalloc d_A returned error code" << endl;
        exit(1);
    }

	cudaMemcpy(ey_gpu, ey, sizeof(DATA_TYPE) * (NX + 1) * NY, cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
    {
        cout << "cudaMalloc d_A returned error code" << endl;
        exit(1);
    }

	cudaMemcpy(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
    if (cudaGetLastError() != cudaSuccess)
    {
        cout << "cudaMalloc d_A returned error code" << endl;
        exit(1);
    }

    //TODO: subject to change
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid( (size_t)ceil(((float)NY) / ((float)block.x)),
            (size_t)ceil(((float)NX) / ((float)block.y)));

    // without hyperq or graph
    // TODO could use two streams to overlap execution
    // TODO
/*
#ifdef TASK_GRAPH
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);
    if (cudaGetLastError() != cudaSuccess) {
        cout << "can't create graph, error code " << cudaGetLastError() << endl;
    }
    cudaGraphNode_t graph_node_1;
    cudaKernelNodeParams node_1_params;
    node_1_params.blockDim = block;
    node_1_params.gridDim = grid;
    node_1_params.func = kernel1;
    void *params_1[5] = {&_fict_, &ex_gpu, &ey_gpu, &hz_gpu, &t};
    node_1_params.kernelParams = &params_1[0]; 
    cudaGraphAddKernelNode(&graph_node_1, graph, NULL, 0, &node_1_params);

    cudaGraphNode_t graph_node_2;
    cudaKernelNodeParams node_2_params;
    node_2_params.blockDim = block;
    node_2_params.gridDim = grid;
    node_2_params.func = kernel2;
    void *params_2[4] = {&ex_gpu, &ey_gpu, &hz_gpu, &t};
    node_2_params.kernelParams = &params_2[0]; 
    cudaGraphAddKernelNode(&graph_node_2, graph, NULL, 0, &node_2_params);

    cudaGraphNode_t graph_node_3;
    cudaKernelNodeParams node_3_params;
    node_3_params.blockDim = block;
    node_3_params.gridDim = grid;
    node_3_params.func = kernel3;
    void *params_3[4] = {&ex_gpu, &ey_gpu, &hz_gpu, &t};
    node_3_params.kernelParams = &params_3[0];
    cudaGraphNode_t dependencies[2] = {graph_node_1, graph_node_2},
    cudaGraphAddKernelNode(&graph_node_3, graph, dependencies, 2, &node_3_params);

    // construct dependencies
    cudaGraphExec_t pGraphExec;
    cudaGraphInstantiate (&pGraphExec, graph, NULL, NULL, 0);  
#endif
*/
    /*
cudaStream_t streamForGraph;
    CUDA_SAFE_CALL(cudaStreamCreate(&streamForGraph));
    cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
*/
    int t = 0;
    for (; t < tmax; t++) {
        kernel1<<<grid, block>>>(_fict_gpu, ex_gpu, ey_gpu, hz_gpu, t);
        cudaDeviceSynchronize();
        kernel2<<<grid, block>>>(ex_gpu, ey_gpu, hz_gpu, t);
        cudaDeviceSynchronize();
        kernel3<<<grid, block>>>(ex_gpu, ey_gpu, hz_gpu, t);
        cudaDeviceSynchronize();
    }
    /*
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
std::cout << "time: " << milliseconds << std::endl;*/


   /* 
    //<------------------------------------------
    // TODO graph execution
    cudaGraph_t pGraph;
    CUDA_SAFE_CALL(cudaGraphCreate(&pGraph, 0));

    cudaGraphNode_t k1;
    cudaKernelNodeParams node1Params = { 0 };
    void *kernel1Args[5] = {(void *)&_fict_gpu, (void *)&ex_gpu, (void *)&ey_gpu, (void *)&hz_gpu, (void*)&t};
    node1Params.func = (void*)kernel1; // Here I put kernel loaded from ptx
	node1Params.gridDim = grid;
	node1Params.blockDim = block;
	node1Params.sharedMemBytes = 0;
	node1Params.kernelParams = (void **)kernel1Args;
	node1Params.extra = NULL;

    cudaGraphNode_t k2;
    cudaKernelNodeParams node2Params = { 0 };
    void *kernel2Args[4] = {(void *)&ex_gpu, (void *)&ey_gpu, (void *)&hz_gpu, (void *)&t};
    node2Params.func = (void*)kernel2; // Here I put kernel loaded from ptx
	node2Params.gridDim = grid;
	node2Params.blockDim = block;
	node2Params.sharedMemBytes = 0;
	node2Params.kernelParams = (void **)kernel2Args;
	node2Params.extra = NULL;



    cudaGraphNode_t k3;
    cudaKernelNodeParams node3Params = { 0 };
    void *kernel3Args[4] = {(void *)&ex_gpu, (void *)&ey_gpu, (void *)&hz_gpu, (void *)&t};
    node3Params.func = (void*)kernel3; // Here I put kernel loaded from ptx
	node3Params.gridDim = grid;
	node3Params.blockDim = block;
	node3Params.sharedMemBytes = 0;
	node3Params.kernelParams = (void **)kernel3Args;
	node3Params.extra = NULL;



    CUDA_SAFE_CALL(cudaGraphAddKernelNode(&k1, pGraph, NULL, 0, &node1Params));
    CUDA_SAFE_CALL(cudaGraphAddKernelNode(&k2, pGraph, NULL, 0, &node2Params));
    CUDA_SAFE_CALL(cudaGraphAddKernelNode(&k3, pGraph, NULL, 0, &node3Params));
    
    CUDA_SAFE_CALL(cudaGraphAddDependencies(pGraph, &k1, &k3, 1));
    CUDA_SAFE_CALL(cudaGraphAddDependencies(pGraph, &k2, &k3, 1));
    
    cudaGraphExec_t graphExec;
    CUDA_SAFE_CALL(cudaGraphInstantiate(&graphExec, pGraph, NULL, NULL, 0));


    cudaGraphNode_t *nodes = NULL;
    size_t numNodes = 0;
    cudaGraphGetNodes(pGraph, nodes, &numNodes);
    std::cout << "num nodes " << numNodes << std::endl;
    //int version = 0;
    //cudaRuntimeGetVersion(&version);
    //std::cout << "version" << version << std::endl;

    cudaStream_t streamForGraph;
    CUDA_SAFE_CALL(cudaStreamCreate(&streamForGraph));
    cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
    for (int a = 0; a < 500; a++) {
        CUDA_SAFE_CALL(cudaGraphLaunch(graphExec, streamForGraph));
    }


    // TODO ::
    CUDA_SAFE_CALL(cudaStreamSynchronize(streamForGraph));
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
std::cout << "time: " << milliseconds << std::endl;

    //free nodes
    //<---------------------------------------------
*/
    cudaMemcpy(hz_from_gpu, hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyDeviceToHost);
    if (cudaGetLastError() != cudaSuccess) {
        cout << "can't copy results back to host, error code " << cudaGetLastError() << endl;
        exit(1);
    }
    //CUDA_SAFE_CALL(cudaGraphExecDestroy(graphExec));
    //CUDA_SAFE_CALL(cudaStreamDestroy(streamForGraph));
    cudaFree(_fict_gpu);
    cudaFree(ex_gpu);
    cudaFree(ey_gpu);
    cudaFree(hz_gpu);
}

void RunBenchmark(ResultDatabase &DB, OptionParser &op) {
    cout << "Running FDTD" << endl;
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    if (deviceProp.computeMode == cudaComputeModeProhibited) {
        cout << "Error: deivce in <Compute Mode Prohibited>,"
            << "no threads can use ::cudaSetDevice()." << endl;
        exit(1);
    }
    if (cudaGetLastError() != cudaSuccess) {
        cout << "cudaGetDeviceProperties returns error code" << endl;
    }
    cudaSetDevice(DEFAULT_GPU);


    bool quiet = op.getOptionBool("quiet");
    int passes = op.getOptionInt("passes");

    // allocating resources
    DATA_TYPE *_fict_cpu = NULL;
    DATA_TYPE *ex_cpu = NULL;
    DATA_TYPE *ey_cpu = NULL;
    DATA_TYPE *hz_cpu = NULL;
    DATA_TYPE *hz_from_gpu = NULL;

    // alloc without unified mem
    _fict_cpu = (DATA_TYPE *)malloc(tmax * sizeof(DATA_TYPE));
    ex_cpu = (DATA_TYPE *)malloc(NX * (NY + 1) * sizeof(DATA_TYPE));
    ey_cpu = (DATA_TYPE *)malloc((NX + 1) * NY * sizeof(DATA_TYPE));
    hz_cpu = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
    cout << "alloced about " << NX << " bytes" << endl;
    hz_from_gpu = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
    
    srand(1);
    init_arrays(_fict_cpu, ex_cpu, ey_cpu, hz_cpu);

    int pass = 0;
    for (; pass < 1; pass++) {
        run_fdtd_cuda(_fict_cpu, ex_cpu, ey_cpu, hz_cpu, hz_from_gpu);
    }

    // TODO may not be necessary
    srand(1);
    init_arrays(_fict_cpu, ex_cpu, ey_cpu, hz_cpu);

    for (pass = 0; pass < 1; pass ++) {
        run_fdtd_cpu(_fict_cpu, ex_cpu, ey_cpu, hz_cpu);
    }
    compareResults(hz_cpu, hz_from_gpu);
    // clean up
    free(_fict_cpu);
    free(ex_cpu);
    free(ey_cpu);
    free(hz_cpu);
    free(hz_from_gpu);
}
