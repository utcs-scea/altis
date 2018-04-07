#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <limits.h>

#include "cudacommon.h"
#include "OptionParser.h"
#include "ResultDatabase.h"
#include "needle.h"
#include "needle_kernel.cu"

#define LIMIT -999
#define SEED 7

int max_rows, max_cols, penalty;

void runTest(ResultDatabase &resultDB, OptionParser &op);

int blosum62[24][24] = {{4,  -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1,
                         -1, -2, -1, 1,  0, -3, -2, 0, -2, -1, 0,  -4},
                        {-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,
                         -1, -3, -2, -1, -1, -3, -2, -3, -1, 0,  -1, -4},
                        {-2, 0,  6,  1, -3, 0,  0,  0,  1, -3, -3, 0,
                         -2, -3, -2, 1, 0,  -4, -2, -3, 3, 0,  -1, -4},
                        {-2, -2, 1,  6, -3, 0,  2,  -1, -1, -3, -4, -1,
                         -3, -3, -1, 0, -1, -4, -3, -3, 4,  1,  -1, -4},
                        {0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3,
                         -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
                        {-1, 1,  0,  0, -3, 5,  2,  -2, 0, -3, -2, 1,
                         0,  -3, -1, 0, -1, -2, -1, -2, 0, 3,  -1, -4},
                        {-1, 0,  0,  2, -4, 2,  5,  -2, 0, -3, -3, 1,
                         -2, -3, -1, 0, -1, -3, -2, -2, 1, 4,  -1, -4},
                        {0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2,
                         -3, -3, -2, 0,  -2, -2, -3, -3, -1, -2, -1, -4},
                        {-2, 0,  1,  -1, -3, 0,  0, -2, 8, -3, -3, -1,
                         -2, -1, -2, -1, -2, -2, 2, -3, 0, 0,  -1, -4},
                        {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3,
                         1,  0,  -3, -2, -1, -3, -1, 3,  -3, -3, -1, -4},
                        {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2,
                         2,  0,  -3, -2, -1, -2, -1, 1,  -4, -3, -1, -4},
                        {-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,
                         -1, -3, -1, 0,  -1, -3, -2, -2, 0,  1,  -1, -4},
                        {-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1,
                         5,  0,  -2, -1, -1, -1, -1, 1,  -3, -1, -1, -4},
                        {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3,
                         0,  6,  -4, -2, -2, 1,  3,  -1, -3, -3, -1, -4},
                        {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1,
                         -2, -4, 7,  -1, -1, -4, -3, -2, -2, -1, -2, -4},
                        {1,  -1, 1,  0, -1, 0,  0,  0,  -1, -2, -2, 0,
                         -1, -2, -1, 4, 1,  -3, -2, -2, 0,  0,  0,  -4},
                        {0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1,
                         -1, -2, -1, 1,  5,  -2, -2, 0,  -1, -1, 0,  -4},
                        {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3,
                         -1, 1,  -4, -3, -2, 11, 2,  -3, -4, -3, -2, -4},
                        {-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2,
                         -1, 3,  -3, -2, -2, 2,  7,  -1, -3, -2, -1, -4},
                        {0, -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2,
                         1, -1, -2, -2, 0,  -3, -1, 4,  -3, -2, -1, -4},
                        {-2, -1, 3,  4, -3, 0,  1,  -1, 0, -3, -4, 0,
                         -3, -3, -2, 0, -1, -4, -3, -3, 4, 1,  -1, -4},
                        {-1, 0,  0,  1, -3, 3,  4,  -2, 0, -3, -3, 1,
                         -1, -3, -1, 0, -1, -3, -2, -2, 1, 4,  -1, -4},
                        {0,  -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1,
                         -1, -1, -2, 0,  0,  -2, -1, -1, -1, -1, -1, -4},
                        {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
                         -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1}};

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
  op.addOption("dimensions", OPT_INT, "0", "dimensions");
  op.addOption("penalty", OPT_INT, "10", "penalty");
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

  int dim = op.getOptionInt("dimensions");
  penalty = op.getOptionInt("penalty");

  if(penalty < 0) {
      printf("Error: Penalty must be a positive number.\n");
      return;
  }
  if(dim < 0) {
      printf("Error: Dimensions must be positive.\n");
      return;
  }

  if (dim == 0) {
    int problemSizes[4] = {100, 1000, 10000, 100000};
    dim = problemSizes[op.getOptionInt("size") - 1];
  }

  long long num_items = (long long)dim * (long long)dim;
  if(num_items >= INT_MAX) {
      printf("Error: Total size cannot exceed INT_MAX");
      return;
  }

  printf("WG size of kernel = %d \n", BLOCK_SIZE);
  printf("Max Rows: %d\n", dim);
  printf("Max Columns: %d\n", dim);
  printf("Penalty: %d\n", penalty);
  srand(SEED);

  int passes = op.getOptionInt("passes");
  for (int i = 0; i < passes; i++) {
    printf("Pass %d: ", i);
    max_rows = dim;
    max_cols = dim;
    runTest(resultDB, op);
    printf("Done.\n");
  }
}

void runTest(ResultDatabase &resultDB, OptionParser &op) {
  bool verbose = op.getOptionBool("verbose");

  int *input_itemsets, *output_itemsets, *referrence;
  int *matrix_cuda, *referrence_cuda;
  int size;

  max_rows = max_rows + 1;
  max_cols = max_cols + 1;
  referrence = (int *)malloc(max_rows * max_cols * sizeof(int));
  input_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));
  output_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));

  if (!input_itemsets) {
      fprintf(stderr, "Error: Can not allocate memory\n");
  }

  for (int i = 0; i < max_cols; i++) {
    for (int j = 0; j < max_rows; j++) {
      input_itemsets[i * max_cols + j] = 0;
    }
  }

  if(verbose) {
    printf("Start Needleman-Wunsch\n");
  }

  for (int i = 1; i < max_rows; i++) {  // please define your own sequence.
    input_itemsets[i * max_cols] = rand() % 10 + 1;
  }
  for (int j = 1; j < max_cols; j++) {  // please define your own sequence.
    input_itemsets[j] = rand() % 10 + 1;
  }

  for (int i = 1; i < max_cols; i++) {
    for (int j = 1; j < max_rows; j++) {
      referrence[i * max_cols + j] =
          blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];
    }
  }

  for (int i = 1; i < max_rows; i++)
    input_itemsets[i * max_cols] = -i * penalty;
  for (int j = 1; j < max_cols; j++) input_itemsets[j] = -j * penalty;

  size = max_cols * max_rows;
  CUDA_SAFE_CALL(cudaMalloc((void **)&referrence_cuda, sizeof(int) * size));
  CUDA_SAFE_CALL(cudaMalloc((void **)&matrix_cuda, sizeof(int) * size));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;
  double transferTime = 0.;
  double kernelTime = 0;

  cudaEventRecord(start, 0);
  CUDA_SAFE_CALL(cudaMemcpy(referrence_cuda, referrence, sizeof(int) * size,
          cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(matrix_cuda, input_itemsets, sizeof(int) * size,
          cudaMemcpyHostToDevice));
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3; // convert to seconds

  dim3 dimGrid;
  dim3 dimBlock(BLOCK_SIZE, 1);
  int block_width = (max_cols - 1) / BLOCK_SIZE;

  if(verbose) {
      printf("Processing top-left matrix\n");
  }
  // process top-left matrix
  for (int i = 1; i <= block_width; i++) {
    dimGrid.x = i;
    dimGrid.y = 1;
    cudaEventRecord(start, 0);
    needle_cuda_shared_1<<<dimGrid, dimBlock>>>(
            referrence_cuda, matrix_cuda, max_cols, penalty, i, block_width);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();
  }
  if(verbose) {
      printf("Processing bottom-right matrix\n");
  }
  // process bottom-right matrix
  for (int i = block_width - 1; i >= 1; i--) {
    dimGrid.x = i;
    dimGrid.y = 1;
    cudaEventRecord(start, 0);
    needle_cuda_shared_2<<<dimGrid, dimBlock>>>(
        referrence_cuda, matrix_cuda, max_cols, penalty, i, block_width);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    kernelTime += elapsedTime * 1.e-3;
    CHECK_CUDA_ERROR();
  }

  cudaEventRecord(start, 0);
  cudaMemcpy(output_itemsets, matrix_cuda, sizeof(int) * size,
          cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3; // convert to seconds

  string resultsfile = op.getOptionString("resultsfile");
  if(resultsfile != "") {
      FILE *fpo = fopen(resultsfile.c_str(), "w");
      if(verbose) {
        fprintf(fpo, "Print traceback value GPU to %s:\n", resultsfile.c_str());
      }

      for (int i = max_rows - 2, j = max_rows - 2; i >= 0, j >= 0;) {
          int nw, n, w, traceback;
          if (i == max_rows - 2 && j == max_rows - 2) {
              // print the first element
              fprintf(fpo, "%d ", output_itemsets[i*max_cols+j]);
          }
          if (i == 0 && j == 0) {
              break;
          }
          if (i > 0 && j > 0) {
              nw = output_itemsets[(i - 1) * max_cols + j - 1];
              w = output_itemsets[i * max_cols + j - 1];
              n = output_itemsets[(i - 1) * max_cols + j];
          } else if (i == 0) {
              nw = n = LIMIT;
              w = output_itemsets[i * max_cols + j - 1];
          } else if (j == 0) {
              nw = w = LIMIT;
              n = output_itemsets[(i - 1) * max_cols + j];
          } else {
          }

          // traceback = maximum(nw, w, n);
          int new_nw, new_w, new_n;
          new_nw = nw + referrence[i * max_cols + j];
          new_w = w - penalty;
          new_n = n - penalty;

          traceback = maximum(new_nw, new_w, new_n);
          if (traceback == new_nw) {
              traceback = nw;
          }
          if (traceback == new_w) {
              traceback = w;
          }
          if (traceback == new_n) {
              traceback = n;
          }

          fprintf(fpo, "%d ", traceback);
          if (traceback == nw) {
              i--;
              j--;
              continue;
          } else if (traceback == w) {
              j--;
              continue;
          } else if (traceback == n) {
              i--;
              continue;
          } else {
          }
      }

      fclose(fpo);
  }

  cudaFree(referrence_cuda);
  cudaFree(matrix_cuda);

  free(referrence);
  free(input_itemsets);
  free(output_itemsets);

  if(verbose) {
      printf("TransferTime: %f\n", transferTime);
      printf("KernelTime: %f\n", kernelTime);
  }

  char tmp[32];
  sprintf(tmp, "%ditems", size);
  string atts = string(tmp);
  resultDB.AddResult("NW-TransferTime", atts, "sec", transferTime);
  resultDB.AddResult("NW-KernelTime", atts, "sec", kernelTime);
  resultDB.AddResult("NW-TotalTime", atts, "sec", transferTime + kernelTime);
  resultDB.AddResult("NW-Rate_Parity", atts, "N", transferTime / kernelTime);
}
