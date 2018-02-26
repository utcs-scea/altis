#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Utility.h"
#include "cudacommon.h"

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define HALO 1  // halo width along one direction when advancing to the next iteration
#define M_SEED 9

void run(ResultDatabase &resultDB, OptionParser &op);

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
  op.addOption("logfile", OPT_STRING, "", "file to write results to");
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
      int rowSizes[4] = {1, 8, 48, 96};
      int colSizes[4] = {1, 8, 48, 96};
      int pyramidSizes[4] = {20, 40, 60, 80};
      rows = rowSizes[op.getOptionInt("size") - 1] * 1024;
      cols = colSizes[op.getOptionInt("size") - 1] * 1024;
      pyramid_height = pyramidSizes[op.getOptionInt("size") - 1];
  } else {
      rows = rowLen;
      cols = colLen;
      pyramid_height = pyramidHeight;
  }

  printf("Row length: %d\n", rows);
  printf("Column length: %d\n", cols);
  printf("Pyramid height: %d\n", pyramid_height);

  int passes = op.getOptionInt("passes");
  for(int i = 0; i < passes; i++) {
    printf("Pass %d: ", i);
    run(resultDB, op);
    printf("Done.\n");
  }
}

void init(OptionParser &op) {
  data = new int[rows * cols];
  wall = new int *[rows];
  for (int n = 0; n < rows; n++) wall[n] = data + cols * n;
  result = new int[cols];

  int seed = M_SEED;
  srand(seed);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      wall[i][j] = rand() % 10;
    }
  }
  string logfile = op.getOptionString("logfile");
  if(logfile != "") {
    std::fstream fs;
    fs.open(logfile.c_str(), std::fstream::in);
    for (int i = 0; i < rows; i++) {
      fs << "***WALL***" << std::endl;
      for (int j = 0; j < cols; j++) {
        fs << wall[i][j] << " ";
      }
      fs << std::endl;
    }
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
  int blkX = small_block_cols * bx - border;
  int blkXmax = blkX + BLOCK_SIZE - 1;

  // calculate the global thread coordination
  int xidx = blkX + tx;

  // effective range within this block that falls within
  // the valid range of the input data
  // used to rule out computation outside the boundary.
  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1)
                                       : BLOCK_SIZE - 1;

  int W = tx - 1;
  int E = tx + 1;

  W = (W < validXmin) ? validXmin : W;
  E = (E > validXmax) ? validXmax : E;

  bool isValid = IN_RANGE(tx, validXmin, validXmax);

  if (IN_RANGE(xidx, 0, cols - 1)) {
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

  int src = 1, dst = 0;
  for (int t = 0; t < rows - 1; t += pyramid_height) {
    int temp = src;
    src = dst;
    dst = temp;
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
  return dst;
}

void run(ResultDatabase &resultDB, OptionParser &op) {
  init(op);

  /* --------------- pyramid parameters --------------- */
  int borderCols = (pyramid_height)*HALO;
  int smallBlockCol = BLOCK_SIZE - (pyramid_height)*HALO * 2;
  int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

  bool verbose = op.getOptionBool("verbose");

  if(verbose) {
    printf(
        "pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: "
        "%d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
        pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);
  }

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


  string logfile = op.getOptionString("logfile");
  if(!logfile.empty()) {
    std::fstream fs;
    fs.open(logfile.c_str(), std::fstream::app);
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

  string testName = "Pathfinder-";
  resultDB.AddResult(testName+"Transfer_Time", toString(rows) + "x" + toString(cols), "sec", transferTime);
  resultDB.AddResult(testName+"Kernel_Time", toString(rows) + "x" + toString(cols), "sec", kernelTime);
  resultDB.AddResult(testName+"Rate_Parity", toString(rows) + "x" + toString(cols), "N", transferTime/kernelTime);
}
