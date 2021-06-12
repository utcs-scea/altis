//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./../lavaMD.h"								// (in the main program folder)	needed to recognized input parameters

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./../util/timer/timer.h"					// (in library path specified to compiler)	needed by timer
#include "cudacommon.h"

//======================================================================================================================================================150
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_cuda_wrapper.h"				// (in the current directory)

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel_gpu_cuda.cu"						// (in the current directory)	GPU kernel, cannot include with header file because of complications with passing of constant memory variables

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

/// <summary>	An enum constant representing the void option. </summary>
void 
kernel_gpu_cuda_wrapper(par_str par_cpu,
						dim_str dim_cpu,
						box_str* box_cpu,
						FOUR_VECTOR* rv_cpu,
						fp* qv_cpu,
						FOUR_VECTOR* fv_cpu,
                        ResultDatabase &resultDB,
						OptionParser &op)
{
	bool uvm = op.getOptionBool("uvm");

    float kernelTime = 0.0f;
    float transferTime = 0.0f;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float elapsedTime;

	//======================================================================================================================================================150
	//	CPU VARIABLES
	//======================================================================================================================================================150

	//======================================================================================================================================================150
	//	GPU SETUP
	//======================================================================================================================================================150

	//====================================================================================================100
	//	INITIAL DRIVER OVERHEAD
	//====================================================================================================100

	checkCudaErrors(cudaDeviceSynchronize());

	//====================================================================================================100
	//	VARIABLES
	//====================================================================================================100

	box_str* d_box_gpu;
	FOUR_VECTOR* d_rv_gpu;
	fp* d_qv_gpu;
	FOUR_VECTOR* d_fv_gpu;

	dim3 threads;
	dim3 blocks;

	//====================================================================================================100
	//	EXECUTION PARAMETERS
	//====================================================================================================100

	blocks.x = dim_cpu.number_boxes;
	blocks.y = 1;
	threads.x = NUMBER_THREADS;											// define the number of threads in the block
	threads.y = 1;

	//======================================================================================================================================================150
	//	GPU MEMORY				(MALLOC)
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY IN
	//====================================================================================================100

	//==================================================50
	//	boxes
	//==================================================50

	if (uvm) {
		d_box_gpu = box_cpu;
	} else {
		checkCudaErrors(cudaMalloc(	(void **)&d_box_gpu,
					dim_cpu.box_mem));
	}

	//==================================================50
	//	rv
	//==================================================50

	if (uvm) {
		d_rv_gpu = rv_cpu;
	} else {
		checkCudaErrors(cudaMalloc(	(void **)&d_rv_gpu, 
					dim_cpu.space_mem));
	}

	//==================================================50
	//	qv
	//==================================================50

	if (uvm) {
		d_qv_gpu = qv_cpu;
	} else {
		checkCudaErrors(cudaMalloc(	(void **)&d_qv_gpu,
					dim_cpu.space_mem2));
	}

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY
	//====================================================================================================100

	//==================================================50
	//	fv
	//==================================================50

	if (uvm) {
		d_fv_gpu = fv_cpu;
	} else {
		checkCudaErrors(cudaMalloc(	(void **)&d_fv_gpu, 
					dim_cpu.space_mem));
	}

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY IN
	//====================================================================================================100

	//==================================================50
	//	boxes
	//==================================================50

    checkCudaErrors(cudaEventRecord(start, 0));

	if (uvm) {
		// Demand paging
	} else {
		checkCudaErrors(cudaMemcpy(	d_box_gpu, 
					box_cpu,
					dim_cpu.box_mem, 
					cudaMemcpyHostToDevice));
	}

	//==================================================50
	//	rv
	//==================================================50
	
	if (uvm) {
		// Demand paging
	} else {
		checkCudaErrors(cudaMemcpy(	d_rv_gpu,
					rv_cpu,
					dim_cpu.space_mem,
					cudaMemcpyHostToDevice));
	}

	//==================================================50
	//	qv
	//==================================================50

	if (uvm) {
		// Demand paging
	} else {
		checkCudaErrors(cudaMemcpy(	d_qv_gpu,
					qv_cpu,
					dim_cpu.space_mem2,
					cudaMemcpyHostToDevice));
	}

	//====================================================================================================100
	//	GPU MEMORY				(MALLOC) COPY
	//====================================================================================================100

	//==================================================50
	//	fv
	//==================================================50

	if (uvm) {
		// Demand paging
	} else {
		checkCudaErrors(cudaMemcpy(	d_fv_gpu, 
					fv_cpu, 
					dim_cpu.space_mem, 
					cudaMemcpyHostToDevice));
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    transferTime += elapsedTime * 1.e-3;

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150

	// launch kernel - all boxes
    checkCudaErrors(cudaEventRecord(start, 0));
	kernel_gpu_cuda<<<blocks, threads>>>(	par_cpu,
											dim_cpu,
											d_box_gpu,
											d_rv_gpu,
											d_qv_gpu,
											d_fv_gpu);
	checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    kernelTime += elapsedTime * 1.e-3;

    CHECK_CUDA_ERROR();
	checkCudaErrors(cudaDeviceSynchronize());

	//======================================================================================================================================================150
	//	GPU MEMORY			COPY (CONTD.)kernel
	//======================================================================================================================================================150

    checkCudaErrors(cudaEventRecord(start, 0));

	if (uvm) {
		checkCudaErrors(cudaMemPrefetchAsync(d_fv_gpu, dim_cpu.space_mem, cudaCpuDeviceId));
        checkCudaErrors(cudaStreamSynchronize(0));
	} else {
		checkCudaErrors(cudaMemcpy(	fv_cpu, 
					d_fv_gpu,
					dim_cpu.space_mem, 
					cudaMemcpyDeviceToHost));
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    transferTime += elapsedTime * 1.e-3;

    char atts[1024];
    sprintf(atts, "boxes1d:%d", dim_cpu.boxes1d_arg);
    resultDB.AddResult("lavamd_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("lavamd_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("lavamd_parity", atts, "N", transferTime / kernelTime);

	//======================================================================================================================================================150
	//	GPU MEMORY DEALLOCATION
	//======================================================================================================================================================150

	if (uvm) {
		// Demand paging, no need to free
	} else {
		checkCudaErrors(cudaFree(d_rv_gpu));
		checkCudaErrors(cudaFree(d_qv_gpu));
		checkCudaErrors(cudaFree(d_fv_gpu));
		checkCudaErrors(cudaFree(d_box_gpu));
	}
}
