#include "fdtd.h"

// TODO subject to change
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

using namespace std;

void addBenchmarkSpecOptions(OptionParser *op) {
   // TODO, maybe add benchmark specs 
}

void init_arrays(DATA_TYPE *_fict_, DATA_TYPE *ex,
        DATA_TYPE *ey, DATA_TYPE *hz) {
    assert(_fict_ && ex && ey && hz);

    int i = 0
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

void run_fdtd_cuda(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, DATA_TYPE *hz_from_gpu) {
    assert(_fict_ && ex && ey && hz && hz_from_gpu);

#ifndef UNIFIED_MEMORY
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

    int t = 0;
    for (; t < tmax; t++) {
        kernel1<<<grid, block>>>(_fict_gpu, ex_gpu, ey_gpu, hz_gpu, t);
    }

#else

#endif

}

void RunBenchmark(ResultDatabase &result DB, OptionParser &op) {
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

#ifndef UNIFIED_MEMORY
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
    hz_from_gpu = (DATA_TYPE *)malloc(NX * NY * sizeof(DATA_TYPE));
    srand(1);
    init_arrays(_fict_cpu, ex_cpu, ey_cpu, hz_cpu);

    int pass = 0;
    for (; pass < passes; pass++) {
        run_fdtd_cuda(_fict_cpu, ex_cpu, ey_cpu, hz_cpu, hz_gpu);
    }
#else
    DATA_TYPE *_fict_ = NULL;
    DATA_TYPE *ex = NULL;
    DATA_TYPE *ey = NULL;
    DATA_TYPE *hz_cpu = NULL;
    DATA_TYPE *hz_gpu = NULL;
    cudaMallocManaged(&_fict_, tmax * sizeof(DATA_TYPE));
    if (cudaGetLastError() != cudaSuccess) {
        cout << "cudaMallocManged failed" << endl;
    }
    cudaMallocManaged(&ex, NX * (NY + 1) * sizeof(DATA_TYPE));
    if (cudaGetLastError() != cudaSuccess) {
        cout << "cudaMallocManged failed" << endl;
    }
    cudaMallocManaged(&ey, (NX + 1) * NY * sizeof(DATA_TYPE));
    if (cudaGetLastError() != cudaSuccess) {
        cout << "cudaMallocManged failed" << endl;
    }
    cudaMallocManaged(&hz_cpu, NX * NY * sizeof(DATA_TYPE));
    if (cudaGetLastError() != cudaSuccess) {
        cout << "cudaMallocManged failed" << endl;
    cudaMallocManaged(&hz_gpu, NX * NY * sizeof(DATA_TYPE));
    if (cudaGetLastError() != cudaSuccess) {
        cout << "cudaMallocManged failed" << endl;
    }
    srand(1);
    init_arrays(_fict_, ex, ey, hz_cpu);

    int pass = 0;
    for (; pass < passes; pass++) {
        run_fdtd_cuda(_fict_, ex, ey, hz_cpu, hz_gpu);
    }
#endif
    

    
}
