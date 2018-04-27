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

void run(int borderCols, int smallBlockCol, int blockCols, ResultDatabase &resultDB, OptionParser &op);

int rows, cols;
int *data;
int **wall;
int *result;
int pyramid_height;

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
  op.addOption("resultsfile", OPT_STRING, "", "file to write results to");
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
// Modifications:
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  int rowLen = op.getOptionInt("rows");
  int colLen = op.getOptionInt("cols");
  int pyramidHeight = op.getOptionInt("pyramidHeight");
  
  if(rowLen == 0 || colLen == 0 || pyramidHeight == 0) {
      printf("Parameters not fully specified, using preset problem size\n");
      int rowSizes[4] = {8, 16, 32, 64};
      int colSizes[4] = {4, 8, 16, 24};
      int pyramidSizes[4] = {2, 4, 8, 16};
      rows = rowSizes[op.getOptionInt("size") - 1];
      cols = colSizes[op.getOptionInt("size") - 1] * 1024 * 1024;
      pyramid_height = pyramidSizes[op.getOptionInt("size") - 1];
  } else {
      rows = rowLen;
      cols = colLen;
      pyramid_height = pyramidHeight;
  }

  printf("Row length: %d\n", rows);
  printf("Column length: %d\n", cols);
  printf("Pyramid height: %d\n", pyramid_height);

  /* --------------- pyramid parameters --------------- */
  int borderCols = (pyramid_height)*HALO;
  int smallBlockCol = BLOCK_SIZE - (pyramid_height)*HALO * 2;
  int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

  printf(
          "gridSize: [%d],border:[%d],blockSize:"
          "[%d],blockGrid:[%d],targetBlock:[%d]\n",
          cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);

  int passes = op.getOptionInt("passes");
  for(int i = 0; i < passes; i++) {
    printf("Pass %d: ", i);
    run(borderCols, smallBlockCol, blockCols, resultDB, op);
    printf("Done.\n");
  }
}

void init(OptionParser &op) {
  data = new int[rows * cols];
  wall = new int *[rows];
  for (int n = 0; n < rows; n++) wall[n] = data + (int)cols * n;
  result = new int[cols];

  srand(SEED);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      wall[i][j] = rand() % 10;
    }
  }
  string resultsfile = op.getOptionString("resultsfile");
  if(resultsfile != "") {
    std::fstream fs;
    fs.open(resultsfile.c_str(), std::fstream::in);
    fs.close();
  }
}

void fatal(char *s) { fprintf(stderr, "error: %s\n", s); }

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

__global__ void dynproc_kernel(int iteration, int *gpuWall, int *gpuSrc,
                               int *gpuResults, int cols, int rows,
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
  int blkX = (int)small_block_cols * (int)bx - (int)border;
  int blkXmax = blkX + (int)BLOCK_SIZE - 1;

  // calculate the global thread coordination
  int xidx = blkX + (int)tx;

  // effective range within this block that falls within
  // the valid range of the input data
  // used to rule out computation outside the boundary.
  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax = (blkXmax > (int)cols - 1) ? (int)BLOCK_SIZE - 1 - (blkXmax - (int)cols + 1)
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

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols,
              int pyramid_height, int blockCols, int borderCols, double& kernelTime) {
  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(blockCols);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  int numStreams = 1;
  cudaStream_t streams[numStreams];
  for(int s = 0; s < numStreams; s++) {
      cudaStreamCreate(&streams[s]);
  }
  int src = 1, dst = 0;
  for (int t = 0; t < rows - 1; t += pyramid_height) {
    for(int s = 0; s < numStreams; s++) {
    int temp = src;
    src = dst;
    dst = temp;

#ifdef HYPERQ
    if(t == 0 && s == 0) {
        cudaEventRecord(start, streams[s]);
    }
    dynproc_kernel<<<dimGrid, dimBlock, 0, streams[s]>>>(
        MIN(pyramid_height, rows - t - 1), gpuWall, gpuResult[src],
        gpuResult[dst], cols, rows, t, borderCols);
    if(t + pyramid_height >= rows - 1 && s == numStreams - 1) {
        cudaDeviceSynchronize();
        cudaEventRecord(stop, streams[s]);
        cudaEventSynchronize(stop);
        CHECK_CUDA_ERROR();
        cudaEventElapsedTime(&elapsedTime, start, stop);
        kernelTime += elapsedTime * 1.e-3;
    }
#else
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
#endif
      }
    }
  return dst;
}

void run(int borderCols, int smallBlockCol, int blockCols, ResultDatabase &resultDB, OptionParser &op) {
  // initialize data
  init(op);

  int *gpuWall, *gpuResult[2];
  int size = rows * cols;

  CUDA_SAFE_CALL(cudaMalloc((void **)&gpuResult[0], sizeof(int) * cols));
  CUDA_SAFE_CALL(cudaMalloc((void **)&gpuResult[1], sizeof(int) * cols));
  CUDA_SAFE_CALL(cudaMalloc((void **)&gpuWall, sizeof(int) * (size - cols)));

  // Cuda events and times
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;
  double transferTime = 0.;
  double kernelTime = 0;

  cudaEventRecord(start, 0);
  CUDA_SAFE_CALL(cudaMemcpy(gpuResult[0], data, sizeof(int) * cols, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(gpuWall, data+cols, sizeof(int) * (size-cols),
             cudaMemcpyHostToDevice));
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3; // convert to seconds

  int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height,
                            blockCols, borderCols, kernelTime);

  cudaEventRecord(start, 0);
  CUDA_SAFE_CALL(cudaMemcpy(result, gpuResult[final_ret], sizeof(int) * cols,
             cudaMemcpyDeviceToHost));
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3; // convert to seconds


  string resultsfile = op.getOptionString("resultsfile");
  if(!resultsfile.empty()) {
    std::fstream fs;
    fs.open(resultsfile.c_str(), std::fstream::app);
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

  cudaFree(gpuWall);
  cudaFree(gpuResult[0]);
  cudaFree(gpuResult[1]);

  delete[] data;
  delete[] wall;
  delete[] result;

  string atts = toString(rows) + "x" + toString(cols);
  resultDB.AddResult("Pathfinder-TransferTime", atts, "sec", transferTime);
  resultDB.AddResult("Pathfinder-KernelTime", atts, "sec", kernelTime);
  resultDB.AddResult("Pathfinder-TotalTime", atts, "sec", transferTime + kernelTime);
  resultDB.AddResult("Pathfinder-Rate_Parity", atts, "N", transferTime/kernelTime);
}
