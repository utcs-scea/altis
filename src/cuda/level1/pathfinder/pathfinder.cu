////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	\altis\src\cuda\level1\pathfinder\pathfinder.cu
//
// summary:	Pathfinder class
// 
// origin: Rodinia (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Utility.h"
#include "cudacommon.h"

#define BLOCK_SIZE 512
#define STR_SIZE 256
#define HALO 1  // halo width along one direction when advancing to the next iteration
#define SEED 7

void run(int borderCols, int smallBlockCol, int blockCols,
         ResultDatabase &resultDB, OptionParser &op);

int rows, cols;
int *data;
int **wall;
int *result;
int pyramid_height;

int device_id;

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing.  The user is allowed to specify
//   the size of the input data in kiB.
//
// Arguments:
//   op: the options parser / parameter database
//
// Programmer: Anthony Danalis
// Creation: September 08, 2009
// Returns:  nothing
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("rows", OPT_INT, "0", "number of rows");
  op.addOption("cols", OPT_INT, "0", "number of cols");
  op.addOption("pyramidHeight", OPT_INT, "0", "pyramid height");
  op.addOption("instances", OPT_INT, "32", "number of pathfinder instances to run");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the pathfinder benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing, results are stored in resultDB
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications: Bodun Hu
// add support for UVM
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
  printf("Running Pathfinder\n");
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  device_id = device;

  bool quiet = op.getOptionInt("quiet");
  int rowLen = op.getOptionInt("rows");
  int colLen = op.getOptionInt("cols");
  int pyramidHeight = op.getOptionInt("pyramidHeight");

  if (rowLen == 0 || colLen == 0 || pyramidHeight == 0) {
    printf("Using preset problem size %d\n", (int)op.getOptionInt("size"));
    int rowSizes[5] = {8, 16, 32, 40, 48};
    int colSizes[5] = {8, 16, 32, 40, 48};
    int pyramidSizes[5] = {4, 8, 16, 32, 36};
    rows = rowSizes[op.getOptionInt("size") - 1] * 1024;
    cols = colSizes[op.getOptionInt("size") - 1] * 1024;
    pyramid_height = pyramidSizes[op.getOptionInt("size") - 1];
  } else {
    rows = rowLen;
    cols = colLen;
    pyramid_height = pyramidHeight;
  }

  if(!quiet) {
      printf("Row length: %d\n", rows);
      printf("Column length: %d\n", cols);
      printf("Pyramid height: %d\n", pyramid_height);
  }

  /* --------------- pyramid parameters --------------- */
  int borderCols = (pyramid_height)*HALO;
  int smallBlockCol = BLOCK_SIZE - (pyramid_height)*HALO * 2;
  int blockCols =
      cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

  if(!quiet) {
      printf("gridSize: [%d],border:[%d],blockSize:[%d],blockGrid:[%d],targetBlock:[%d]\n",
              cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);
  }

  int passes = op.getOptionInt("passes");
  for (int i = 0; i < passes; i++) {
    if(!quiet) {
        printf("Pass %d: ", i);
    }
    run(borderCols, smallBlockCol, blockCols, resultDB, op);
    if(!quiet) {
        printf("Done.\n");
    }
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Initializes parameters for computing.  </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="op">	[in,out] The option specified by the user. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void init(OptionParser &op) {
  const bool uvm = op.getOptionBool("uvm");
  const bool uvm_advise = op.getOptionBool("uvm-advise");
  const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
  const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");

  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    checkCudaErrors(cudaMallocManaged(&data, sizeof(int) * rows * cols));
    checkCudaErrors(cudaMallocManaged(&wall, sizeof(int *) * rows));
    for (int n = 0; n < rows; n++) wall[n] = data + (int)cols * n;
    // checkCudaErrors(cudaMallocManaged(&result, sizeof(int) * cols));
  } else {
    data = new int[rows * cols];
    wall = new int *[rows];
    for (int n = 0; n < rows; n++) wall[n] = data + (int)cols * n;
    result = new int[cols];
  }

  srand(SEED);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      wall[i][j] = rand() % 10;
    }
  }
  string outfile = op.getOptionString("outputFile");
  if (outfile != "") {
    std::fstream fs;
    fs.open(outfile.c_str(), std::fstream::in);
    fs.close();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines whether given point is in range. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="x">  	data point. </param>
/// <param name="min">	The minimum. </param>
/// <param name="max">	The maximum. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that specifies how to clamp range. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="x">  	A void to process. </param>
/// <param name="min">	The minimum. </param>
/// <param name="max">	The maximum. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Dynproc kernel. </summary>
///
/// <remarks>	Bodun Hu (bodunhu@utexas.edu), 1/7/2021. </remarks>
///
/// <param name="iteration"> 	The iteration. </param>
/// <param name="gpuWall">   	[in,out] If non-null, the GPU wall. </param>
/// <param name="gpuSrc">	 	[in,out] If non-null, the GPU source. </param>
/// <param name="gpuResults">	[in,out] If non-null, the GPU results. </param>
/// <param name="cols">		 	The cols. </param>
/// <param name="rows">		 	The rows. </param>
/// <param name="startStep"> 	The start step. </param>
/// <param name="border">	 	The border. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void dynproc_kernel(int iteration, int *gpuWall,
                               int *gpuSrc, int *gpuResults,
                               int cols, int rows,
                               int startStep, int border) {
  __shared__ int prev[BLOCK_SIZE];
  __shared__ int result[BLOCK_SIZE];

  int bx = blockIdx.x;
  int tx = threadIdx.x;

  // each block finally computes result for a small block
  // after N iterations.
  // it is the non-overlapping small blocks that cover
  // all the input data

  // calculate the small block size
  int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

  // calculate the boundary for the block according to
  // the boundary of its small block
  int blkX =
      (int)small_block_cols * (int)bx - (int)border;
  int blkXmax = blkX + (int)BLOCK_SIZE - 1;

  // calculate the global thread coordination
  int xidx = blkX + (int)tx;

  // effective range within this block that falls within
  // the valid range of the input data
  // used to rule out computation outside the boundary.
  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax =
      (blkXmax > (int)cols - 1)
          ? (int)BLOCK_SIZE - 1 - (blkXmax - (int)cols + 1)
          : (int)BLOCK_SIZE - 1;

  int W = tx - 1;
  int E = tx + 1;

  W = (W < validXmin) ? validXmin : W;
  E = (E > validXmax) ? validXmax : E;

  bool isValid = IN_RANGE(tx, validXmin, validXmax);

  if (IN_RANGE(xidx, 0, (int)cols - 1)) {
    prev[tx] = gpuSrc[xidx];
  }
  __syncthreads();  // [Ronny] Added sync to avoid race on prev Aug. 14 2012
  bool computed;
  for (int i = 0; i < iteration; i++) {
    computed = false;
    if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid) {
      computed = true;
      int left = prev[W];
      int up = prev[tx];
      int right = prev[E];
      int shortest = MIN(left, up);
      shortest = MIN(shortest, right);
      int index = cols * (startStep + i) + xidx;
      result[tx] = shortest + gpuWall[index];
    }
    __syncthreads();
    if (i == iteration - 1) break;
    if (computed)  // Assign the computation range
      prev[tx] = result[tx];
    __syncthreads();  // [Ronny] Added sync to avoid race on prev Aug. 14 2012
  }

  // update the global memory
  // after the last iteration, only threads coordinated within the
  // small block perform the calculation and switch on ``computed''
  if (computed) {
    gpuResults[xidx] = result[tx];
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the path. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="gpuWall">		 	[in,out] If non-null, the GPU wall. </param>
/// <param name="gpuResult">	 	[in,out] If non-null, the GPU result. </param>
/// <param name="rows">			 	The rows. </param>
/// <param name="cols">			 	The cols. </param>
/// <param name="pyramid_height">	Height of the pyramid. </param>
/// <param name="blockCols">	 	The block cols. </param>
/// <param name="borderCols">	 	The border cols. </param>
/// <param name="kernelTime">	 	[in,out] The kernel time. </param>
/// <param name="hyperq">		 	True to hyperq. </param>
/// <param name="numStreams">	 	Number of streams. </param>
///
/// <returns>	The calculated path. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

int calc_path(int *gpuWall, int *gpuResult[2], int rows,
                    int cols, int pyramid_height,
                    int blockCols, int borderCols,
                    double &kernelTime, bool hyperq, int numStreams) {
  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(blockCols);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  cudaStream_t streams[numStreams];
  for (int s = 0; s < numStreams; s++) {
    cudaStreamCreate(&streams[s]);
  }
  int src = 1, dst = 0;
  for (int t = 0; t < rows - 1; t += pyramid_height) {
    for (int s = 0; s < numStreams; s++) {
      int temp = src;
      src = dst;
      dst = temp;

    if(hyperq) {
      if (t == 0 && s == 0) {
        cudaEventRecord(start, streams[s]);
      }
      dynproc_kernel<<<dimGrid, dimBlock, 0, streams[s]>>>(
          MIN(pyramid_height, rows - t - 1), gpuWall, gpuResult[src],
          gpuResult[dst], cols, rows, t, borderCols);
      if (t + pyramid_height >= rows - 1 && s == numStreams - 1) {
        cudaDeviceSynchronize();
        cudaEventRecord(stop, streams[s]);
        cudaEventSynchronize(stop);
        CHECK_CUDA_ERROR();
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime += elapsedTime * 1.e-3;
      }
    } else {
      cudaEventRecord(start, 0);
      dynproc_kernel<<<dimGrid, dimBlock>>>(
          MIN(pyramid_height, rows - t - 1), gpuWall, gpuResult[src],
          gpuResult[dst], cols, rows, t, borderCols);
      cudaDeviceSynchronize();
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      CHECK_CUDA_ERROR();
      cudaEventElapsedTime(&elapsedTime, start, stop);
      kernelTime += elapsedTime * 1.e-3;
    }
    }
  }
  return dst;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Runs the calc_path kernel.
/// 			added UVM and hyperQ support </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="borderCols">   	The border cols. </param>
/// <param name="smallBlockCol">	The small block cols. </param>
/// <param name="blockCols">		The block cols. </param>
/// <param name="resultDB">			[in,out] The result database. </param>
/// <param name="op">				[in,out] The options. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void run(int borderCols, int smallBlockCol, int blockCols,
         ResultDatabase &resultDB, OptionParser &op) {
  // initialize data
  init(op);
  const bool uvm = op.getOptionBool("uvm");
  const bool uvm_advise = op.getOptionBool("uvm-advise");
  const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
  const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");

  int *gpuWall, *gpuResult[2];
  int size = rows * cols;

  if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise) {
    gpuResult[0] = data;
    checkCudaErrors(cudaMallocManaged((void **)&gpuResult[1], sizeof(int) * cols));
    checkCudaErrors(cudaMallocManaged((void **)&gpuWall, sizeof(int) * (size - cols)));
    // gpuWall = data + cols;
  } else {
    checkCudaErrors(cudaMalloc((void **)&gpuResult[0], sizeof(int) * cols));
    checkCudaErrors(cudaMalloc((void **)&gpuResult[1], sizeof(int) * cols));
    checkCudaErrors(cudaMalloc((void **)&gpuWall, sizeof(int) * (size - cols)));
  }

  // Cuda events and times
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  float elapsedTime;
  double transferTime = 0.;
  double kernelTime = 0;

  checkCudaErrors(cudaEventRecord(start, 0));

  if (uvm) {
    // do nothing
  } else if (uvm_advise) {
    checkCudaErrors(cudaMemAdvise(gpuResult[0], sizeof(int) * cols, cudaMemAdviseSetPreferredLocation, device_id));
    checkCudaErrors(cudaMemAdvise(gpuWall, sizeof(int) * (size - cols), cudaMemAdviseSetPreferredLocation, device_id));
    checkCudaErrors(cudaMemAdvise(gpuWall, sizeof(int) * (size - cols), cudaMemAdviseSetReadMostly, device_id));
  } else if (uvm_prefetch) {
    checkCudaErrors(cudaMemPrefetchAsync(gpuResult[0], sizeof(int) * cols, device_id));
    cudaStream_t s1;
    checkCudaErrors(cudaStreamCreate(&s1));
    checkCudaErrors(cudaMemPrefetchAsync(gpuWall, sizeof(int) * (size - cols), device_id, s1));
    checkCudaErrors(cudaStreamDestroy(s1));
  } else if (uvm_prefetch_advise) {
    checkCudaErrors(cudaMemAdvise(gpuResult[0], sizeof(int) * cols, cudaMemAdviseSetPreferredLocation, device_id));
    checkCudaErrors(cudaMemAdvise(gpuWall, sizeof(int) * (size - cols), cudaMemAdviseSetPreferredLocation, device_id));
    checkCudaErrors(cudaMemAdvise(gpuWall, sizeof(int) * (size - cols), cudaMemAdviseSetReadMostly, device_id));
    checkCudaErrors(cudaMemPrefetchAsync(gpuResult[0], sizeof(int) * cols, device_id));
    cudaStream_t s1;
    checkCudaErrors(cudaStreamCreate(&s1));
    checkCudaErrors(cudaMemPrefetchAsync(gpuWall, sizeof(int) * (size - cols), device_id, s1));
    checkCudaErrors(cudaStreamDestroy(s1));
  } else {
    checkCudaErrors(cudaMemcpy(gpuResult[0], data, sizeof(int) * cols,
                            cudaMemcpyHostToDevice));
  
    checkCudaErrors(cudaMemcpy(gpuWall, data + cols,
                            sizeof(int) * (size - cols),
                            cudaMemcpyHostToDevice));
  }

  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  transferTime += elapsedTime * 1.e-3;  // convert to seconds

  int instances = op.getOptionInt("instances");

#ifdef HYPERQ
  double hyperqKernelTime = 0;
  /// <summary>	Calc the path with hyperQ enabled. </summary>
  int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height, blockCols,
            borderCols, hyperqKernelTime, true, instances);
#else
  int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height, blockCols,
                borderCols, kernelTime, false, instances);
#endif

  checkCudaErrors(cudaEventRecord(start, 0));

  if (uvm) {
    result = gpuResult[final_ret];
  } else if (uvm_advise) {
    result = gpuResult[final_ret];
    checkCudaErrors(cudaMemAdvise(gpuResult[final_ret], sizeof(int) * cols, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    checkCudaErrors(cudaMemAdvise(gpuResult[final_ret], sizeof(int) * cols, cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
  } else if (uvm_prefetch) {
    result = gpuResult[final_ret];
    checkCudaErrors(cudaMemPrefetchAsync(result, sizeof(int) * cols, cudaCpuDeviceId));
  } else if (uvm_prefetch_advise) {
    result = gpuResult[final_ret];
    checkCudaErrors(cudaMemAdvise(gpuResult[final_ret], sizeof(int) * cols, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    checkCudaErrors(cudaMemAdvise(gpuResult[final_ret], sizeof(int) * cols, cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
    checkCudaErrors(cudaMemPrefetchAsync(result, sizeof(int) * cols, cudaCpuDeviceId));
  } else {
    checkCudaErrors(cudaMemcpy(result, gpuResult[final_ret],
                    sizeof(int) * cols, cudaMemcpyDeviceToHost));
  }

  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  transferTime += elapsedTime * 1.e-3;  // convert to seconds

  /// <summary>	Output the results to a file. </summary>
  string outfile = op.getOptionString("outputFile");
  if (!outfile.empty()) {
    std::fstream fs;
    fs.open(outfile.c_str(), std::fstream::app);
    fs << "***DATA***" << std::endl;
    for (int i = 0; i < cols; i++) {
      fs << data[i] << " ";
    }
    fs << std::endl;
    fs << "***RESULT***" << std::endl;
    for (int i = 0; i < cols; i++) {
      fs << result[i] << " ";
    }
    fs << std::endl;
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /// <summary>	cleanup. </summary>
  ///
  /// <remarks>	Ed, 5/20/2020. </remarks>
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  // cudaFree(gpuWall);
  cudaFree(gpuResult[0]);
  cudaFree(gpuResult[1]);

if (!uvm && !uvm_advise && !uvm_prefetch && !uvm_prefetch_advise) {
  delete[] data;
  delete[] wall;
  delete[] result;
}

  string atts = toString(rows) + "x" + toString(cols);
#ifdef HYPERQ
  /// <summary>	The result db. add result. </summary>
  resultDB.AddResult("pathfinder_hyperq_transfer_time", atts, "sec", transferTime);
  resultDB.AddResult("pathfinder_hyperq_kernel_time", atts, "sec", hyperqKernelTime);
  resultDB.AddResult("pathfinder_hyperq_total_time", atts, "sec", hyperqKernelTime + transferTime);
  resultDB.AddResult("pathfinder_hyperq_parity", atts, "N",
                     transferTime / hyperqKernelTime);
  resultDB.AddResult("pathfinder_hyperq_speedup", atts, "sec",
                     kernelTime/hyperqKernelTime);
#else
  resultDB.AddResult("pathfinder_transfer_time", atts, "sec", transferTime);
  resultDB.AddResult("pathfinder_kernel_time", atts, "sec", kernelTime);
  resultDB.AddResult("pathfinder_total_time", atts, "sec", kernelTime + transferTime);
  resultDB.AddResult("pathfinder_parity", atts, "N",
                     transferTime / kernelTime);

#endif
  resultDB.AddOverall("Time", "sec", kernelTime+transferTime);
}
