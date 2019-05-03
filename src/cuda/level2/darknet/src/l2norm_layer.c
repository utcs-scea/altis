#include "l2norm_layer.h"
#include "activations.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

void test_l2norm_layer_forward(int batch, int input_size) {
    printf("----- l2 norm forward -----\n");
    layer l = make_l2norm_layer(batch, input_size);
    network *net = make_network(1);
    // May consider changing the values here
    l.out_c = 64;
    l.out_w = 224;
    l.out_h = 224;
    net->input_gpu = cuda_make_array(NULL, l.batch * l.inputs);
    forward_l2norm_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("--------------------\n");
}

void test_l2norm_layer_backward(int batch, int input_size) {
    printf("----- l2 norm backward -----\n");
    layer l = make_l2norm_layer(batch, input_size);
    network *net = make_network(1);
    net->delta_gpu =  cuda_make_array(NULL, l.batch * l.inputs);
    l.delta_gpu =  cuda_make_array(NULL, l.batch * l.inputs);
    backward_l2norm_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("--------------------\n");
}

layer make_l2norm_layer(int batch, int inputs)
{
    fprintf(stderr, "l2norm                                         %4d\n",  inputs);
    layer l = {0};
    l.type = L2NORM;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = calloc(inputs*batch, sizeof(float));
    l.scales = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));

    l.forward = forward_l2norm_layer;
    l.backward = backward_l2norm_layer;
    #ifdef GPU
    l.forward_gpu = forward_l2norm_layer_gpu;
    l.backward_gpu = backward_l2norm_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.scales_gpu = cuda_make_array(l.output, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
    #endif
    return l;
}

void forward_l2norm_layer(const layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    l2normalize_cpu(l.output, l.scales, l.batch, l.out_c, l.out_w*l.out_h);
}

void backward_l2norm_layer(const layer l, network net)
{
    //axpy_cpu(l.inputs*l.batch, 1, l.scales, 1, l.delta, 1);
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void forward_l2norm_layer_gpu(const layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    l2normalize_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_w*l.out_h);
}

void backward_l2norm_layer_gpu(const layer l, network net)
{
    printf("1\n");
    axpy_gpu(l.batch*l.inputs, 1, l.scales_gpu, 1, l.delta_gpu, 1);
    printf("2\n");
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}

#endif
