#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Utility.h"
#include "cudacommon.h"
#include "backprop.h"
#include "backprop_cuda.h"

#define SEED 7

using namespace std;

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

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
  op.addOption("layerSize", OPT_INT, "0",
               "specify layer size (must be a multiple of 16)");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the backprop benchmark
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

  setup(resultDB, op);
}

void setup(ResultDatabase &resultDB, OptionParser &op) {
  int layer_size = op.getOptionInt("layerSize");

  // if no layer size specified, default to using a preset problem size
  if(layer_size == 0) {
    int probSizes[4] = {1, 8, 48, 96};
    layer_size = probSizes[op.getOptionInt("size") - 1] * 1024 * 8;
  }

  if (layer_size % 16 != 0) {
    fprintf(stderr, "The number of input points must be divided by 16\n");
    exit(0);
  }
  printf("Input layer size: %d\n", layer_size);

#ifdef CPU
  printf("Performing CPU computation\n");
#endif

#ifdef GPU
  printf("Performing GPU computation\n");
#endif

  int passes = op.getOptionInt("passes");
  for(int i = 0; i < passes; i++) {
    printf("Pass %d: ", i);
    bpnn_initialize(SEED);
    backprop_face(layer_size, resultDB);
    printf("Done.\n");
  }

}

void backprop_face(int layer_size, ResultDatabase &resultDB) {
  BPNN *net;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)

  load(net, layer_size);
  // entering the training kernel, only one iteration
  bpnn_train_cuda(net, &out_err, &hid_err, resultDB);
  bpnn_free(net);
}

void bpnn_train_cuda(BPNN *net, float *eo, float *eh, ResultDatabase &resultDB) {
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

#ifdef GPU
  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  num_blocks = in / 16;
  dim3 grid(1, num_blocks);
  dim3 threads(16, 16);

  input_weights_one_dim = (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
  input_weights_prev_one_dim =
      (float *)malloc((in + 1) * (hid + 1) * sizeof(float));
  partial_sum = (float *)malloc(num_blocks * WIDTH * sizeof(float));

  // this preprocessing stage is added to correct the bugs of wrong memcopy
  // using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {
    for (int j = 0; j <= hid; j++) {
      input_weights_one_dim[m] = net->input_weights[k][j];
      input_weights_prev_one_dim[m] = net->input_prev_weights[k][j];
      m++;
    }
  }

  CUDA_SAFE_CALL(cudaMalloc((void **)&input_cuda, (in + 1) * sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&output_hidden_cuda, (hid + 1) * sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&hidden_partial_sum, num_blocks * WIDTH * sizeof(float)));

#endif

#ifdef CPU

  bpnn_layerforward(net->input_units, net->hidden_units, net->input_weights, in,

#endif

#ifdef GPU

  // Cuda events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;
  double transferTime = 0.;
  double kernelTime = 0;

  // Copy inputs to GPU
  cudaEventRecord(start, 0);
  CUDA_SAFE_CALL(cudaMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float),
             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(input_hidden_cuda, input_weights_one_dim,
             (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice));
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3; // convert to seconds

  // Perform bpnn layer forward
  cudaEventRecord(start, 0);
  bpnn_layerforward_CUDA<<<grid, threads>>>(input_cuda, output_hidden_cuda,
                                            input_hidden_cuda,
                                            hidden_partial_sum, in, hid);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  CHECK_CUDA_ERROR();
  cudaEventElapsedTime(&elapsedTime, start, stop);
  kernelTime += elapsedTime * 1.e-3;

  cudaEventRecord(start, 0);
  cudaMemcpy(partial_sum, hidden_partial_sum,
             num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3; // convert to seconds

  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {
      sum += partial_sum[k * hid + j - 1];
    }
    sum += net->input_weights[0][j];
    net->hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }
#endif

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
                    hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out,
                    &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
                    net->hidden_weights, net->hidden_units, &hid_err);
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
                      net->hidden_weights, net->hidden_prev_weights);

#ifdef CPU

  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
                      net->input_weights, net->input_prev_weights);

#endif

#ifdef GPU

  cudaMalloc((void **)&hidden_delta_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void **)&input_prev_weights_cuda,
             (in + 1) * (hid + 1) * sizeof(float));

  cudaEventRecord(start, 0);
  cudaMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(input_prev_weights_cuda, input_weights_prev_one_dim,
             (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_hidden_cuda, input_weights_one_dim,
             (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3; // convert to seconds

  cudaEventRecord(start, 0);
  bpnn_adjust_weights_cuda<<<grid, threads>>>(hidden_delta_cuda, hid,
                                              input_cuda, in, input_hidden_cuda,
                                              input_prev_weights_cuda);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  CHECK_CUDA_ERROR();
  cudaEventElapsedTime(&elapsedTime, start, stop);
  kernelTime += elapsedTime * 1.e-3;

  cudaEventRecord(start, 0);
  cudaMemcpy(net->input_units, input_cuda, (in + 1) * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(input_weights_one_dim, input_hidden_cuda,
             (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3; // convert to seconds

  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);

  free(partial_sum);
  free(input_weights_one_dim);
  free(input_weights_prev_one_dim);

  string atts = "layersize:" + toString(in);
  resultDB.AddResult("Backprop-Transfer_Time", atts, "sec", transferTime);
  resultDB.AddResult("Backprop-Kernel_Time", atts, "sec", kernelTime);
  resultDB.AddResult("Backprop-Rate_Parity", atts, "N", transferTime/kernelTime);

#endif
}

////////////////////////////////////////////////////////////////////////////////
__global__ void bpnn_layerforward_CUDA(float *input_cuda,
                                       float *output_hidden_cuda,
                                       float *input_hidden_cuda,
                                       float *hidden_partial_sum, int in,
                                       int hid) {
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
  int index_in = HEIGHT * by + ty + 1;

  __shared__ float input_node[HEIGHT];
  __shared__ float weight_matrix[HEIGHT][WIDTH];

  if (tx == 0)
    input_node[ty] = input_cuda[index_in];
  __syncthreads();
  weight_matrix[ty][tx] = input_hidden_cuda[index];
  __syncthreads();
  weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];
  __syncthreads();

  for (int i = 1; i <= __log2f(HEIGHT); i++) {
    int power_two = __powf(2, i);
    if (ty % power_two == 0)
      weight_matrix[ty][tx] =
          weight_matrix[ty][tx] + weight_matrix[ty + power_two / 2][tx];
    __syncthreads();
  }

  input_hidden_cuda[index] = weight_matrix[ty][tx];
  __syncthreads();

  if (tx == 0) {
    hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
  }
}

__global__ void bpnn_adjust_weights_cuda(float *delta, int hid, float *ly,
                                         int in, float *w, float *oldw) {
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
  int index_y = HEIGHT * by + ty + 1;
  int index_x = tx + 1;

  w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
  oldw[index] =
      ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

  __syncthreads();

  if (ty == 0 && by == 0) {
    w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
    oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
  }
}
////////////////////////////////////////////////////////////////////////////////

