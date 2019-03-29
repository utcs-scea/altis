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

#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define DEFAULT_GPU 0

/* Problem size */
#define tmax 500
#define NX 2048
#define NY 2048
#define SMALL_FLOAT_VAL 0.00000001f

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
# ifndef DATA_TYPE
#  define DATA_TYPE float
# endif

using namespace std;

void addBenchmarkSpecOptions(OptionParser &op) {
   // TODO, maybe add benchmark specs 
}

float absVal(float a)
{
  if (a<0) return (a * -1);
  else     return a;
}

float percentDiff(double val1, double val2)
{
  if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01)) return 0.0f;
  else                                                return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
}


void init_arrays(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
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

void runFdtd(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz)
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
	cout << "Non-Matching CPU-GPU Outputs Beyond Error Threshold of "
        << PERCENT_DIFF_ERROR_THRESHOLD <<  " percent: " << fail << endl;
}

__global__ void fdtd_step1_kernel(DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
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

__global__ void fdtd_step2_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ((i < NX) && (j < NY) && (j > 0))
	{
		ex[i * (NY+1) + j] = ex[i * (NY+1) + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
	}
}

__global__ void fdtd_step3_kernel(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NX) && (j < NY))
	{
		hz[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * (NY+1) + (j+1)] - ex[i * (NY+1) + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
	}
}

void fdtdCuda(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz, DATA_TYPE* hz_outputFromGpu)
{
    cudaError_t error;
	double t_start, t_end;

	DATA_TYPE *_fict_gpu;
	DATA_TYPE *ex_gpu;
	DATA_TYPE *ey_gpu;
	DATA_TYPE *hz_gpu;

	error=cudaMalloc((void **)&_fict_gpu, sizeof(DATA_TYPE) * tmax);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMalloc((void **)&ex_gpu, sizeof(DATA_TYPE) * NX * (NY + 1));
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMalloc((void **)&ey_gpu, sizeof(DATA_TYPE) * (NX + 1) * NY);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMalloc((void **)&hz_gpu, sizeof(DATA_TYPE) * NX * NY);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }


	error=cudaMemcpy(_fict_gpu, _fict_, sizeof(DATA_TYPE) * tmax, cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(ex_gpu, ex, sizeof(DATA_TYPE) * NX * (NY + 1), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(ey_gpu, ey, sizeof(DATA_TYPE) * (NX + 1) * NY, cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	error=cudaMemcpy(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid( (size_t)ceil(((float)NY) / ((float)block.x)), (size_t)ceil(((float)NX) / ((float)block.y)));


	for(int t = 0; t< tmax; t++)
	{
		fdtd_step1_kernel<<<grid,block>>>(_fict_gpu, ex_gpu, ey_gpu, hz_gpu, t);
		//cudaThreadSynchronize();
		fdtd_step2_kernel<<<grid,block>>>(ex_gpu, ey_gpu, hz_gpu, t);
		cudaThreadSynchronize();
		fdtd_step3_kernel<<<grid,block>>>(ex_gpu, ey_gpu, hz_gpu, t);
		cudaThreadSynchronize();
	}

	error=cudaMemcpy(hz_outputFromGpu, hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
        {
            printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }

	cudaFree(_fict_gpu);
	cudaFree(ex_gpu);
	cudaFree(ey_gpu);
	cudaFree(hz_gpu);
}

void RunBenchmark(ResultDatabase &DB, OptionParser &op)
{
  
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

  /* Prepare ctuning vars */
  long ct_repeat=0;
  long ct_repeat_max=1;

  double t_start, t_end;

  DATA_TYPE* _fict_;
  DATA_TYPE* ex;
  DATA_TYPE* ey;
  DATA_TYPE* hz;
  DATA_TYPE* hz_outputFromGpu;


  /* Run kernel. */
  //if (getenv("CT_REPEAT_MAIN")!=NULL) ct_repeat_max=atol(getenv("CT_REPEAT_MAIN"));

  _fict_ = (DATA_TYPE*)malloc(tmax*sizeof(DATA_TYPE));
  ex = (DATA_TYPE*)malloc(NX*(NY+1)*sizeof(DATA_TYPE));
  ey = (DATA_TYPE*)malloc((NX+1)*NY*sizeof(DATA_TYPE));
  hz = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
  hz_outputFromGpu = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));

  srand(1);
  init_arrays(_fict_, ex, ey, hz);

  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    fdtdCuda(_fict_, ex, ey, hz, hz_outputFromGpu);
  }

  srand(1);
  init_arrays(_fict_, ex, ey, hz);

  for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
  {
    runFdtd(_fict_, ex, ey, hz);
  }

  compareResults(hz, hz_outputFromGpu);

  free(_fict_);
  free(ex);
  free(ey);
  free(hz);
  free(hz_outputFromGpu);
}
