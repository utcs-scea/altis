#include "rnn_layer.h"
#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define cudnnErrCheck(stat) {cudnnErrCheck_((stat), __FILE__, __LINE__);}
void cudnnErrCheck_(cudnnStatus_t stat, const char *file, int line) {
   if (stat != CUDNN_STATUS_SUCCESS) {
      fprintf(stderr, "cuDNN Error: %s %s %d\n", cudnnGetErrorString(stat), file, line);
   }
}

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}


void test_rnn_layer_forward(int batch, int hiddenSize, int outputs, int seqLength,
        int numLayers) {
    
    network *net = make_network(1);
    layer l = make_rnn_layer(batch, hiddenSize, outputs, seqLength, numLayers);

    forward_rnn_layer_gpu(l, *net);
}


static void increment_layer(layer *l, int steps)
{
    int num = l->outputs*l->batch*steps;
    l->output += num;
    l->delta += num;
    l->x += num;
    l->x_norm += num;

#ifdef GPU
    l->output_gpu += num;
    l->delta_gpu += num;
    l->x_gpu += num;
    l->x_norm_gpu += num;
#endif
}

layer make_rnn_layer(int batch, int hiddenSize, int outputs, int seqLength, int numLayers)
{
    fprintf(stderr, "RNN Layer: %d inputs, %d outputs\n", hiddenSize, outputs);
    layer l = {0};
    l.batch = batch;
    l.type = RNN;
    l.hiddenSize = hiddenSize;
    l.inputs = hiddenSize;
    l.seqLength = seqLength;
    l.numLayers = numLayers;
    l.outputs = outputs;

    l.x_gpu = cuda_make_array(0, l.seqLength * l.inputs * l.batch);
    l.hx = cuda_make_array(0, l.numLayers * l.hiddenSize * batch);
    l.cx = cuda_make_array(0, l.numLayers * l.hiddenSize * batch);
    
    l.dx = cuda_make_array(0, l.seqLength * l.inputs * l.batch);
    l.dhx = cuda_make_array(0, l.numLayers * l.hiddenSize * batch);
    l.dcx = cuda_make_array(0, l.numLayers * l.hiddenSize * batch);

    l.output_gpu = cuda_make_array(0, l.seqLength * l.hiddenSize * l.batch);
    l.hy = cuda_make_array(0, l.numLayers * l.hiddenSize * batch);
    l.cy = cuda_make_array(0, l.numLayers * l.hiddenSize * batch);

    l.dy = cuda_make_array(0, l.seqLength * l.hiddenSize * l.batch);
    l.dhy = cuda_make_array(0, l.numLayers * l.hiddenSize * batch);
    l.dcy = cuda_make_array(0, l.numLayers * l.hiddenSize * batch);


    l.xDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
    l.yDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
    l.dxDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
    l.dyDesc = (cudnnTensorDescriptor_t*)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));

    int dimA[3];
    int strideA[3];

    for (int i = 0; i < l.seqLength; i++) {
        cudnnErrCheck(cudnnCreateTensorDescriptor(&(l.xDesc[i])));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&(l.yDesc[i])));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&(l.dxDesc[i])));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&(l.dyDesc[i])));

        dimA[0] = batch;
        dimA[1] = l.inputs;
        dimA[2] = 1;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        cudnnErrCheck(cudnnSetTensorNdDescriptor((l.xDesc)[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor((l.dxDesc)[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));

        dimA[0] = l.batch;
        dimA[1] = l.hiddenSize;
        dimA[2] = 1;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        cudnnErrCheck(cudnnSetTensorNdDescriptor((l.yDesc)[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor((l.dyDesc)[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
    }

    dimA[0] = l.numLayers;
    dimA[1] = l.batch;
    dimA[2] = l.hiddenSize;

    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;

    cudnnErrCheck(cudnnCreateTensorDescriptor(&l.hxDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&l.cxDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&l.hyDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&l.cyDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&l.dhxDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&l.dcxDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&l.dhyDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&l.dcyDesc));

    cudnnErrCheck(cudnnSetTensorNdDescriptor(l.hxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(l.cxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(l.hyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(l.cyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(l.dhxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(l.dcxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(l.dhyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(l.dcyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

    unsigned long long seed = 213ull;

    cudnnErrCheck(cudnnCreateDropoutDescriptor(&l.dropoutDesc));
    cudnnErrCheck(cudnnDropoutGetStatesSize(cudnn_handle(), &l.dropoutRandStatesSizeInBytes));
    l.dropoutRandStates = cuda_make_array(0, l.dropoutRandStatesSizeInBytes); // may devide here
    cudnnErrCheck(cudnnSetDropoutDescriptor(l.dropoutDesc,
                             cudnn_handle(),
                             0.4,
                             l.dropoutRandStates,
                             l.dropoutRandStatesSizeInBytes,
                             seed));

    cudnnErrCheck(cudnnCreateRNNDescriptor(&l.rnnDesc)); 

    // TODO may need to change the type to allow more options here
    l.RNNMode = CUDNN_LSTM;
    l.algo = CUDNN_RNN_ALGO_STANDARD;
    cudnnErrCheck(cudnnSetRNNDescriptor_v6(cudnn_handle(),
                                       l.rnnDesc,
                                       l.hiddenSize,
                                       l.numLayers,
                                       l.dropoutDesc,
                                       CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
                                       CUDNN_UNIDIRECTIONAL,
                                       l.RNNMode,
                                       l.algo, //CUDNN_RNN_ALGO_STANDARD,
                                       CUDNN_DATA_FLOAT));
    
    // setup parameters
    cudnnErrCheck(cudnnCreateFilterDescriptor(&l.wDesc));
    cudnnErrCheck(cudnnCreateFilterDescriptor(&l.dwDesc));
    cudnnErrCheck(cudnnGetRNNParamsSize(cudnn_handle(), l.rnnDesc, (l.xDesc)[0],
                &l.weightsSize, CUDNN_DATA_FLOAT));

    int dimW[3];
    dimW[0] = l.weightsSize / sizeof(float);
    dimW[1] = 1;
    dimW[2] = 1;

    cudnnErrCheck(cudnnSetFilterNdDescriptor(l.wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));
    cudnnErrCheck(cudnnSetFilterNdDescriptor(l.dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));

    cudaErrCheck(cudaMalloc((void**)&l.weights_gpu, l.weightsSize));
    cudaErrCheck(cudaMalloc((void**)&l.dw, l.weightsSize));

    cudnnErrCheck(cudnnGetRNNWorkspaceSize(cudnn_handle(), l.rnnDesc, l.seqLength,
                l.xDesc, &l.workSize));
    cudnnErrCheck(cudnnGetRNNTrainingReserveSize(cudnn_handle(), l.rnnDesc,
                l.seqLength, l.xDesc, &l.reserveSize));

    cudaErrCheck(cudaMalloc((void**)&l.workspace, l.workSize));
    cudaErrCheck(cudaMalloc((void**)&l.reserveSpace, l.reserveSize));

    // Ignore weight init for now
    cudaErrCheck(cudaDeviceSynchronize());

#ifdef GPU
    l.forward_gpu = forward_rnn_layer_gpu;
    l.backward_gpu = backward_rnn_layer_gpu;
    l.update_gpu = update_rnn_layer_gpu;
    //l.state_gpu = cuda_make_array(0, batch*outputs);
    //l.prev_state_gpu = cuda_make_array(0, batch*outputs);
    //l.output_gpu = l.output_layer->output_gpu;
    //l.delta_gpu = l.output_layer->delta_gpu;
#ifdef CUDNN
#endif
#endif

    return l;
}

#ifdef GPU

void pull_rnn_layer(layer l)
{
    pull_connected_layer(*(l.input_layer));
    pull_connected_layer(*(l.self_layer));
    pull_connected_layer(*(l.output_layer));
}

void push_rnn_layer(layer l)
{
    push_connected_layer(*(l.input_layer));
    push_connected_layer(*(l.self_layer));
    push_connected_layer(*(l.output_layer));
}

void update_rnn_layer_gpu(layer l, update_args a)
{
    update_connected_layer_gpu(*(l.input_layer),  a);
    update_connected_layer_gpu(*(l.self_layer),   a);
    update_connected_layer_gpu(*(l.output_layer), a);
}

void forward_rnn_layer_gpu(layer l, network net)
{
    network s = {0};
    s.train = net.train;

    cudnnErrCheck(cudnnRNNForwardTraining(cudnn_handle(),
                                         l.rnnDesc,
                                         l.seqLength,
                                         l.xDesc,
                                         l.x_gpu,
                                         l.hxDesc,
                                         l.hx,
                                         l.cxDesc,
                                         l.cx,
                                         l.wDesc,
                                         l.weights_gpu,
                                         l.yDesc,
                                         l.output_gpu,
                                         l.hyDesc,
                                         l.hy,
                                         l.cyDesc,
                                         l.cy,
                                         l.workspace,
                                         l.workSize,
                                         l.reserveSpace,
                                         l.reserveSize));
}

void backward_rnn_layer_gpu(layer l, network net)
{
    network s = {0};
    s.train = net.train;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);
    increment_layer(&input_layer,  l.steps - 1);
    increment_layer(&self_layer,   l.steps - 1);
    increment_layer(&output_layer, l.steps - 1);
    float *last_input = input_layer.output_gpu;
    float *last_self = self_layer.output_gpu;
    for (i = l.steps-1; i >= 0; --i) {
        fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);

        s.input_gpu = l.state_gpu;
        s.delta_gpu = self_layer.delta_gpu;
        backward_connected_layer_gpu(output_layer, s);

        if(i != 0) {
            fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, input_layer.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, self_layer.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
        }else {
            copy_gpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.state_gpu, 1);
        }

        copy_gpu(l.outputs*l.batch, self_layer.delta_gpu, 1, input_layer.delta_gpu, 1);

        s.input_gpu = l.state_gpu;
        s.delta_gpu = (i > 0) ? self_layer.delta_gpu - l.outputs*l.batch : 0;
        if (i == 0) s.delta_gpu = 0;
        backward_connected_layer_gpu(self_layer, s);

        s.input_gpu = net.input_gpu + i*l.inputs*l.batch;
        if(net.delta_gpu) s.delta_gpu = net.delta_gpu + i*l.inputs*l.batch;
        else s.delta_gpu = 0;
        backward_connected_layer_gpu(input_layer, s);

        increment_layer(&input_layer,  -1);
        increment_layer(&self_layer,   -1);
        increment_layer(&output_layer, -1);
    }
    fill_gpu(l.outputs * l.batch, 0, l.state_gpu, 1);
    axpy_gpu(l.outputs * l.batch, 1, last_input, 1, l.state_gpu, 1);
    axpy_gpu(l.outputs * l.batch, 1, last_self, 1, l.state_gpu, 1);
}
#endif
