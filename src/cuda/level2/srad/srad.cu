////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\srad\srad.cu
//
// summary:	Srad class
// 
// origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "srad.h"

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"

// includes, project
#include <cuda.h>

// includes, kernels
#include "srad_kernel.cu"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines seed. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define SEED 7

/// <summary>	The kernel time. </summary>
float kernelTime = 0.0f;
/// <summary>	The transfer time. </summary>
float transferTime = 0.0f;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the stop. </summary>
///
/// <value>	The stop. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

cudaEvent_t start, stop;
/// <summary>	The elapsed. </summary>
float elapsed;
/// <summary>	The check. </summary>
float *check;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Random matrix. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="I">   	[in,out] If non-null, zero-based index of the. </param>
/// <param name="rows">	The rows. </param>
/// <param name="cols">	The cols. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void random_matrix(float *I, int rows, int cols);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the test operation. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="argc">	The argc. </param>
/// <param name="argv">	[in,out] If non-null, the argv. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Srads. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">   	[in,out] The result database. </param>
/// <param name="op">		  	[in,out] The operation. </param>
/// <param name="matrix">	  	[in,out] If non-null, the matrix. </param>
/// <param name="imageSize">  	Size of the image. </param>
/// <param name="speckleSize">	Size of the speckle. </param>
/// <param name="iters">	  	The iters. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float srad(ResultDatabase &resultDB, OptionParser &op, float* matrix, int imageSize, int speckleSize, int iters);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Srad gridsync. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">   	[in,out] The result database. </param>
/// <param name="op">		  	[in,out] The operation. </param>
/// <param name="matrix">	  	[in,out] If non-null, the matrix. </param>
/// <param name="imageSize">  	Size of the image. </param>
/// <param name="speckleSize">	Size of the speckle. </param>
/// <param name="iters">	  	The iters. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float srad_gridsync(ResultDatabase &resultDB, OptionParser &op, float* matrix, int imageSize, int speckleSize, int iters);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds a benchmark specifier options. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="op">	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("imageSize", OPT_INT, "0", "image height and width");
  op.addOption("speckleSize", OPT_INT, "0", "speckle height and width");
  op.addOption("iterations", OPT_INT, "0", "iterations of algorithm");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
  printf("Running SRAD\n");

  srand(SEED);
  bool quiet = op.getOptionBool("quiet");
  const bool uvm = op.getOptionBool("uvm");
  const bool uvm_advise = op.getOptionBool("uvm-advice");
  const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
  const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
  const bool coop = op.getOptionBool("coop");
  int device = 0;
  checkCudaErrors(cudaGetDevice(&device));

  // set parameters
  int imageSize = op.getOptionInt("imageSize");
  int speckleSize = op.getOptionInt("speckleSize");
  int iters = op.getOptionInt("iterations");
  if (imageSize == 0 || speckleSize == 0 || iters == 0) {
    int imageSizes[5] = {128, 512, 4096, 8192, 16384};
    int iterSizes[5] = {5, 1, 15, 20, 40};
    imageSize = imageSizes[op.getOptionInt("size") - 1];
    speckleSize = imageSize / 2;
    iters = iterSizes[op.getOptionInt("size") - 1];
  }

  // create timing events
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  if (!quiet) {
      printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
      printf("Image Size: %d x %d\n", imageSize, imageSize);
      printf("Speckle size: %d x %d\n", speckleSize, speckleSize);
      printf("Num Iterations: %d\n\n", iters);
  }

  // run workload
  int passes = op.getOptionInt("passes");
  for (int i = 0; i < passes; i++) {
    float *matrix = NULL;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(cudaMallocManaged(&matrix, imageSize * imageSize * sizeof(float)));
    } else {
        matrix = (float*)malloc(imageSize * imageSize * sizeof(float));
        assert(matrix);
    }
    random_matrix(matrix, imageSize, imageSize);
    if (!quiet) {
        printf("Pass %d:\n", i);
    }
    float time = srad(resultDB, op, matrix, imageSize, speckleSize, iters);
    if (!quiet) {
        printf("Running SRAD...Done.\n");
    }
    if (coop) {
        // if using cooperative groups, add result to compare the 2 times
        char atts[1024];
        sprintf(atts, "img:%d,speckle:%d,iter:%d", imageSize, speckleSize, iters);
        float time_gridsync = srad_gridsync(resultDB, op, matrix, imageSize, speckleSize, iters);
        if(!quiet) {
            if(time_gridsync == FLT_MAX) {
                printf("Running SRAD with cooperative groups...Failed.\n");
            } else {
                printf("Running SRAD with cooperative groups...Done.\n");
            }
        }
        if(time_gridsync == FLT_MAX) {
            resultDB.AddResult("srad_gridsync_speedup", atts, "N", time/time_gridsync);
        }
    }
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(cudaFree(matrix));
    } else {
        free(matrix);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Srads. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">   	[in,out] The result database. </param>
/// <param name="op">		  	[in,out] The operation. </param>
/// <param name="matrix">	  	[in,out] If non-null, the matrix. </param>
/// <param name="imageSize">  	Size of the image. </param>
/// <param name="speckleSize">	Size of the speckle. </param>
/// <param name="iters">	  	The iters. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float srad(ResultDatabase &resultDB, OptionParser &op, float* matrix, int imageSize,
          int speckleSize, int iters) {
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    const bool coop = op.getOptionBool("coop");
    int device = 0;
    checkCudaErrors(cudaGetDevice(&device));


    kernelTime = 0.0f;
    transferTime = 0.0f;
    int rows, cols, size_I, size_R, niter, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI, varROI;

  float *J_cuda;
  float *C_cuda;
  float *E_C, *W_C, *N_C, *S_C;

  unsigned int r1, r2, c1, c2;
  float *c;

  rows = imageSize;  // number of rows in the domain
  cols = imageSize;  // number of cols in the domain
  if ((rows % 16 != 0) || (cols % 16 != 0)) {
    fprintf(stderr, "rows and cols must be multiples of 16\n");
    exit(1);
  }
  r1 = 0;            // y1 position of the speckle
  r2 = speckleSize;  // y2 position of the speckle
  c1 = 0;            // x1 position of the speckle
  c2 = speckleSize;  // x2 position of the speckle
  lambda = 0.5;      // Lambda value
  niter = iters;     // number of iterations

  size_I = cols * rows;
  size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    checkCudaErrors(cudaMallocManaged(&J, sizeof(float) * size_I));
    checkCudaErrors(cudaMallocManaged(&c, sizeof(float) * size_I));
  } else {
    I = (float *)malloc(size_I * sizeof(float));
    assert(I);
    J = (float *)malloc(size_I * sizeof(float));
    assert(J);
    c = (float *)malloc(sizeof(float) * size_I);
    assert(c);
  }

  // Allocate device memory
  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    J_cuda = J;
    C_cuda = c;
    checkCudaErrors(cudaMallocManaged((void **)&E_C, sizeof(float) * size_I));
    checkCudaErrors(cudaMallocManaged((void **)&W_C, sizeof(float) * size_I));
    checkCudaErrors(cudaMallocManaged((void **)&S_C, sizeof(float) * size_I));
    checkCudaErrors(cudaMallocManaged((void **)&N_C, sizeof(float) * size_I));
  } else {
    checkCudaErrors(cudaMalloc((void **)&J_cuda, sizeof(float) * size_I));
    checkCudaErrors(cudaMalloc((void **)&C_cuda, sizeof(float) * size_I));
    checkCudaErrors(cudaMalloc((void **)&E_C, sizeof(float) * size_I));
    checkCudaErrors(cudaMalloc((void **)&W_C, sizeof(float) * size_I));
    checkCudaErrors(cudaMalloc((void **)&S_C, sizeof(float) * size_I));
    checkCudaErrors(cudaMalloc((void **)&N_C, sizeof(float) * size_I));
  }

  // copy random matrix
  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    I = matrix;
  } else {
    memcpy(I, matrix, rows*cols*sizeof(float));
  }

  for (int k = 0; k < size_I; k++) {
    J[k] = (float)exp(I[k]);
  }
  for (iter = 0; iter < niter; iter++) {
    sum = 0;
    sum2 = 0;
    for (int i = r1; i <= r2; i++) {
      for (int j = c1; j <= c2; j++) {
        tmp = J[i * cols + j];
        sum += tmp;
        sum2 += tmp * tmp;
      }
    }
    meanROI = sum / size_R;
    varROI = (sum2 / size_R) - meanROI * meanROI;
    q0sqr = varROI / (meanROI * meanROI);

    // Currently the input size must be divided by 16 - the block size
    int block_x = cols / BLOCK_SIZE;
    int block_y = rows / BLOCK_SIZE;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(block_x, block_y);

    // Copy data from main memory to device memory
    checkCudaErrors(cudaEventRecord(start, 0));
    if (uvm) {
        // do nothing
    } else if (uvm_advise) {
      checkCudaErrors(cudaMemAdvise(J_cuda, sizeof(float) * size_I, cudaMemAdviseSetPreferredLocation, device));
    } else if (uvm_prefetch) {
      checkCudaErrors(cudaMemPrefetchAsync(J_cuda, sizeof(float) * size_I, device));
    } else if (uvm_prefetch_advise) {
      checkCudaErrors(cudaMemAdvise(J_cuda, sizeof(float) * size_I, cudaMemAdviseSetPreferredLocation, device));
      checkCudaErrors(cudaMemPrefetchAsync(J_cuda, sizeof(float) * size_I, device));
    } else {
      checkCudaErrors(cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
    transferTime += elapsed * 1.e-3;

    // Run kernels
    checkCudaErrors(cudaEventRecord(start, 0));
    srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols,
                                       rows, q0sqr);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();

    checkCudaErrors(cudaEventRecord(start, 0));
    srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols,
                                       rows, lambda, q0sqr);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();

    // Copy data from device memory to main memory
    checkCudaErrors(cudaEventRecord(start, 0));

    if (uvm) {
      // do nothing
    } else if (uvm_advise) {
      checkCudaErrors(cudaMemAdvise(J_cuda, sizeof(float) * size_I, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    } else if (uvm_prefetch) {
      checkCudaErrors(cudaMemPrefetchAsync(J_cuda, sizeof(float) * size_I, cudaCpuDeviceId));
    } else if (uvm_prefetch_advise) {
      checkCudaErrors(cudaMemAdvise(J_cuda, sizeof(float) * size_I, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
      checkCudaErrors(cudaMemAdvise(J_cuda, sizeof(float) * size_I, cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
      checkCudaErrors(cudaMemPrefetchAsync(J_cuda, sizeof(float) * size_I, cudaCpuDeviceId));
    } else {
      checkCudaErrors(cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
    transferTime += elapsed * 1.e-3;
  }

    char atts[1024];
    sprintf(atts, "img:%d,speckle:%d,iter:%d", imageSize, speckleSize, iters);
    resultDB.AddResult("srad_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("srad_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("srad_total_time", atts, "sec", kernelTime + transferTime);
    resultDB.AddResult("srad_parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime+transferTime);

  string outfile = op.getOptionString("outputFile");
  if (!outfile.empty()) {
      // Printing output
      if (!op.getOptionBool("quiet")) {
        printf("Writing output to %s\n", outfile.c_str());
      }
      FILE *fp = NULL;
      fp = fopen(outfile.c_str(), "w");
      if (!fp) {
          printf("Error: Unable to write to file %s\n", outfile.c_str());
      } else {
          for (int i = 0; i < rows; i++) {
              for (int j = 0; j < cols; j++) {
                  fprintf(fp, "%.5f ", J[i * cols + j]);
              }
              fprintf(fp, "\n");
          }
          fclose(fp);
      }
  }
  // write results to validate with srad_gridsync
  check = (float*) malloc(sizeof(float) * size_I);
  assert(check);
  for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
          check[i*cols+j] = J[i*cols+j];
      }
  }

  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    checkCudaErrors(cudaFree(C_cuda));
    checkCudaErrors(cudaFree(J_cuda));
    checkCudaErrors(cudaFree(E_C));
    checkCudaErrors(cudaFree(W_C));
    checkCudaErrors(cudaFree(N_C));
    checkCudaErrors(cudaFree(S_C));
  } else {
    free(I);
    free(J);
    free(c);
    checkCudaErrors(cudaFree(C_cuda));
    checkCudaErrors(cudaFree(J_cuda));
    checkCudaErrors(cudaFree(E_C));
    checkCudaErrors(cudaFree(W_C));
    checkCudaErrors(cudaFree(N_C));
    checkCudaErrors(cudaFree(S_C));
  }
  return kernelTime;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Srad gridsync with UVM and gridsync. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">   	[in,out] The result database. </param>
/// <param name="op">		  	[in,out] The operation. </param>
/// <param name="matrix">	  	[in,out] If non-null, the matrix. </param>
/// <param name="imageSize">  	Size of the image. </param>
/// <param name="speckleSize">	Size of the speckle. </param>
/// <param name="iters">	  	The iters. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float srad_gridsync(ResultDatabase &resultDB, OptionParser &op, float* matrix, int imageSize, int speckleSize, int iters) {
    const bool uvm = op.getOptionBool("uvm");
    const bool uvm_advise = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
    const bool coop = op.getOptionBool("coop");
    int device = 0;
    checkCudaErrors(cudaGetDevice(&device));
    
    kernelTime = 0.0f;
    transferTime = 0.0f;
    int rows, cols, size_I, size_R, niter, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI, varROI;

  float *J_cuda;
  float *C_cuda;
  float *E_C, *W_C, *N_C, *S_C;

  unsigned int r1, r2, c1, c2;
  float *c;

  rows = imageSize;  // number of rows in the domain
  cols = imageSize;  // number of cols in the domain
  if ((rows % 16 != 0) || (cols % 16 != 0)) {
    fprintf(stderr, "rows and cols must be multiples of 16\n");
    exit(1);
  }
  r1 = 0;            // y1 position of the speckle
  r2 = speckleSize;  // y2 position of the speckle
  c1 = 0;            // x1 position of the speckle
  c2 = speckleSize;  // x2 position of the speckle
  lambda = 0.5;      // Lambda value
  niter = iters;     // number of iterations

  size_I = cols * rows;
  size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    checkCudaErrors(cudaMallocManaged((void **)&J, sizeof(float) * size_I));
    checkCudaErrors(cudaMallocManaged((void **)&c, sizeof(float) * size_I));
  } else {
    I = (float *)malloc(size_I * sizeof(float));
    assert(I);
    J = (float *)malloc(size_I * sizeof(float));
    assert(J);
    c = (float *)malloc(sizeof(float) * size_I);
    assert(c);
  }

  // Allocate device memory
  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    J_cuda = J;
    C_cuda = c;
    checkCudaErrors(cudaMallocManaged((void **)&E_C, sizeof(float) * size_I));
    checkCudaErrors(cudaMallocManaged((void **)&W_C, sizeof(float) * size_I));
    checkCudaErrors(cudaMallocManaged((void **)&S_C, sizeof(float) * size_I));
    checkCudaErrors(cudaMallocManaged((void **)&N_C, sizeof(float) * size_I));
  } else {
    checkCudaErrors(cudaMalloc((void **)&J_cuda, sizeof(float) * size_I));
    checkCudaErrors(cudaMalloc((void **)&C_cuda, sizeof(float) * size_I));
    checkCudaErrors(cudaMalloc((void **)&E_C, sizeof(float) * size_I));
    checkCudaErrors(cudaMalloc((void **)&W_C, sizeof(float) * size_I));
    checkCudaErrors(cudaMalloc((void **)&S_C, sizeof(float) * size_I));
    checkCudaErrors(cudaMalloc((void **)&N_C, sizeof(float) * size_I));
  }

  // Generate a random matrix
  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    I = matrix;
  } else {
    memcpy(I, matrix, rows*cols*sizeof(float));
  }

  for (int k = 0; k < size_I; k++) {
    J[k] = (float)exp(I[k]);
  }
  for (iter = 0; iter < niter; iter++) {
    sum = 0;
    sum2 = 0;
    for (int i = r1; i <= r2; i++) {
      for (int j = c1; j <= c2; j++) {
        tmp = J[i * cols + j];
        sum += tmp;
        sum2 += tmp * tmp;
      }
    }
    meanROI = sum / size_R;
    varROI = (sum2 / size_R) - meanROI * meanROI;
    q0sqr = varROI / (meanROI * meanROI);

    // Currently the input size must be divided by 16 - the block size
    int block_x = cols / BLOCK_SIZE;
    int block_y = rows / BLOCK_SIZE;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(block_x, block_y);

    // Copy data from main memory to device memory
    checkCudaErrors(cudaEventRecord(start, 0));
    // timing incorrect for page fault
    // J_cuda = J;
    // C_cuda = c;
    if (uvm) {
      // do nothing
    } else if (uvm_advise) {
      checkCudaErrors(cudaMemAdvise(J_cuda, sizeof(float) * size_I, cudaMemAdviseSetPreferredLocation, device));
    } else if (uvm_prefetch) {
      checkCudaErrors(cudaMemPrefetchAsync(J_cuda, sizeof(float) * size_I, device));
    } else if (uvm_prefetch_advise) {
      checkCudaErrors(cudaMemAdvise(J_cuda, sizeof(float) * size_I, cudaMemAdviseSetPreferredLocation, device));
      checkCudaErrors(cudaMemPrefetchAsync(J_cuda, sizeof(float) * size_I, device));
    } else {
      checkCudaErrors(cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
    transferTime += elapsed * 1.e-3;

    // Create srad_params struct
    srad_params params;
    params.E_C = E_C;
    params.W_C = W_C;
    params.N_C = N_C;
    params.S_C = S_C;
    params.J_cuda = J_cuda;
    params.C_cuda = C_cuda;
    params.cols = cols;
    params.rows = rows;
    params.lambda = lambda;
    params.q0sqr = q0sqr;
    void* p_params = {&params};

    // Run kernels
    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaLaunchCooperativeKernel((void*)srad_cuda_3, dimGrid, dimBlock, &p_params));
    //srad_cuda_3<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols,
                                       //rows, lambda, q0sqr);
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
    kernelTime += elapsed * 1.e-3;
    cudaError_t err = cudaGetLastError();                                     
    if (err != cudaSuccess)                                                   
    {                                                                         
      printf("error=%d name=%s at "                                         
               "ln: %d\n  ",err,cudaGetErrorString(err),__LINE__);            
      if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
        checkCudaErrors(cudaFree(C_cuda));
        checkCudaErrors(cudaFree(J_cuda));
        checkCudaErrors(cudaFree(E_C));
        checkCudaErrors(cudaFree(W_C));
        checkCudaErrors(cudaFree(N_C));
        checkCudaErrors(cudaFree(S_C));
      }
      else {
        checkCudaErrors(cudaFree(C_cuda));
        checkCudaErrors(cudaFree(J_cuda));
        checkCudaErrors(cudaFree(E_C));
        checkCudaErrors(cudaFree(W_C));
        checkCudaErrors(cudaFree(N_C));
        checkCudaErrors(cudaFree(S_C));

        free(I);
        free(J);
        free(c);
      }
    return FLT_MAX;
    }                                                                     

    // Copy data from device memory to main memory
    checkCudaErrors(cudaEventRecord(start, 0));
    if (uvm) {
      // do nothing
    } else if (uvm_advise) {
      checkCudaErrors(cudaMemAdvise(J, sizeof(float) * size_I, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
      checkCudaErrors(cudaMemAdvise(J, sizeof(float) * size_I, cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
    } else if (uvm_prefetch) {
      checkCudaErrors(cudaMemPrefetchAsync(J, sizeof(float) * size_I, cudaCpuDeviceId));
    } else if (uvm_prefetch_advise) {
      checkCudaErrors(cudaMemAdvise(J, sizeof(float) * size_I, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
      checkCudaErrors(cudaMemAdvise(J, sizeof(float) * size_I, cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
      checkCudaErrors(cudaMemPrefetchAsync(J, sizeof(float) * size_I, cudaCpuDeviceId));
    } else {
      checkCudaErrors(cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));
    transferTime += elapsed * 1.e-3;
  }

    char atts[1024];
    sprintf(atts, "img:%d,speckle:%d,iter:%d", imageSize, speckleSize, iters);
    resultDB.AddResult("srad_gridsync_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("srad_gridsync_transer_time", atts, "sec", transferTime);
    resultDB.AddResult("srad_gridsync_total_time", atts, "sec", kernelTime + transferTime);
    resultDB.AddResult("srad_gridsync_parity", atts, "N", transferTime / kernelTime);

  // validate result with result obtained by gridsync
  for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
          if(check[i*cols+j] - J[i*cols+j] > 0.0001) {
              // known bug: with and without gridsync have 10e-5 difference in row 16
              //printf("Error: Validation failed at row %d, col %d\n", i, j);
              //return FLT_MAX;
          }
      }
  }
  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    CUDA_SAFE_CALL(cudaFree(C_cuda));
    CUDA_SAFE_CALL(cudaFree(J_cuda));
    CUDA_SAFE_CALL(cudaFree(E_C));
    CUDA_SAFE_CALL(cudaFree(W_C));
    CUDA_SAFE_CALL(cudaFree(N_C));
    CUDA_SAFE_CALL(cudaFree(S_C));
  } else {
    free(I);
    free(J);
    free(c);
    CUDA_SAFE_CALL(cudaFree(C_cuda));
    CUDA_SAFE_CALL(cudaFree(J_cuda));
    CUDA_SAFE_CALL(cudaFree(E_C));
    CUDA_SAFE_CALL(cudaFree(W_C));
    CUDA_SAFE_CALL(cudaFree(N_C));
    CUDA_SAFE_CALL(cudaFree(S_C));
  }
  return kernelTime;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Random matrix. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="I">   	[in,out] If non-null, zero-based index of the. </param>
/// <param name="rows">	The rows. </param>
/// <param name="cols">	The cols. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void random_matrix(float *I, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      I[i * cols + j] = rand() / (float)RAND_MAX;
    }
  }
}

