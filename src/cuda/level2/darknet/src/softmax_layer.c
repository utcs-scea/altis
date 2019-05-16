#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

void test_softmax_layer_forward(int batch, int input_size, int groups) {
    //softmax_layer l = make_softmax_layer(batch, 810000, 2500);
    printf("----- testingf softmax forward -----\n");
    softmax_layer l = make_softmax_layer(batch, input_size, groups);
    network *net = make_network(1);
    l.spatial = 0;  // For every catagory
    net->input_gpu = cuda_make_array(l.output, l.inputs*batch);
    net->truth = calloc(l.inputs*l.batch, sizeof(float));
    l.noloss = 1;
    net->truth_gpu = cuda_make_array(l.delta, l.inputs*batch);
    forward_softmax_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("------------------------------------\n\n");
}

void test_softmax_layer_backward(int batch, int input_size, int groups) {
    //softmax_layer l = make_softmax_layer(batch, 810000, 2500);
    printf("----- testingf softmax backward -----\n");
    softmax_layer l = make_softmax_layer(batch, input_size, groups);
    network *net = make_network(1);
    net->delta_gpu = cuda_make_array(l.delta, l.inputs*batch);
    backward_softmax_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("-------------------------------------\n\n");
}

softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n",  inputs);
    softmax_layer l = {0};
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;
    #ifdef GPU
    l.forward_gpu = forward_softmax_layer_gpu;
    l.backward_gpu = backward_softmax_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#ifdef CUDNN
    l.softmaxAlgo = CUDNN_SOFTMAX_FAST; // can change it
    l.softmaxMode = CUDNN_SOFTMAX_MODE_INSTANCE;
    cudnnCreateTensorDescriptor(&l.softmaxInputTensorDesc);
    const int dimA[4] = {l.batch, l.inputs, 1, 1};
    const int strideA[4] = {1, 1, 1, 1};
    cudnnSetTensorNdDescriptor(l.softmaxInputTensorDesc, CUDNN_DATA_FLOAT, 4,
            dimA, strideA); 
    cudnnCreateTensorDescriptor(&l.softmaxOutputTensorDesc);
    cudnnSetTensorNdDescriptor(l.softmaxOutputTensorDesc, CUDNN_DATA_FLOAT, 4,
            dimA, strideA);

#endif
    #endif
    return l;
}

void forward_softmax_layer(const softmax_layer l, network net)
{
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        }
    } else {
        softmax_cpu(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
    }

    if(net.truth && !l.noloss){
        softmax_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_softmax_layer(const softmax_layer l, network net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_softmax_layer_output(const softmax_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
}

void forward_softmax_layer_gpu(const softmax_layer l, network net)
{
    /*
    if(l.softmax_tree){
        softmax_tree(net.input_gpu, 1, l.batch, l.inputs, l.temperature, l.output_gpu, *l.softmax_tree);
    } else {
        if(l.spatial){
            softmax_gpu(net.input_gpu, l.c, l.batch*l.c, l.inputs/l.c, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu);
        }else{
            softmax_gpu(net.input_gpu, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output_gpu);
        }
    }
    if(net.truth && !l.noloss){
        softmax_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
        if(l.softmax_tree){
            mask_gpu(l.batch*l.inputs, l.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
            mask_gpu(l.batch*l.inputs, l.loss_gpu, SECRET_NUM, net.truth_gpu, 0);
        }
        cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
    */
    int alpha = 1, beta = 0;
    cudnnSoftmaxForward(cudnn_handle(), l.softmaxAlgo, l.softmaxMode, &alpha,
            l.softmaxInputTensorDesc, net.input_gpu, &beta, l.softmaxOutputTensorDesc,
            l.output_gpu);

}

void backward_softmax_layer_gpu(const softmax_layer layer, network net)
{
    axpy_gpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, net.delta_gpu, 1);
}

#endif
