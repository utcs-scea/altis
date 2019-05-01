#include "activation_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test_activation_layer_forward(int batch, int input_size,
                ACTIVATION actv) {
#ifndef GPU
    printf("Can't test activation layer for GPU, exiting...\n");
    exit(1);
#endif
    printf("----- activation layer forward -----\n");

    layer l = make_activation_layer(batch, input_size, actv);
    network *net = make_network(1);
    net->input_gpu = cuda_make_array(l.output, l.batch* l.inputs);
    forward_activation_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("\n");
}

void test_activation_layer_backward(int batch, int input_size,
                ACTIVATION actv) {
#ifndef GPU
    printf("Can't test activation layer for GPU, exiting...\n");
    exit(1);
#endif
    printf("----- activation layer backward -----\n");

    layer l = make_activation_layer(batch, input_size, actv);
    network *net = make_network(1);
    net->delta_gpu = cuda_make_array(l.output, l.batch * l.inputs);
    backward_activation_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("\n");
}

layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = {0};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = calloc(batch*inputs, sizeof(float*));
    l.delta = calloc(batch*inputs, sizeof(float*));

    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
#ifdef GPU
    l.forward_gpu = forward_activation_layer_gpu;
    l.backward_gpu = backward_activation_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    l.activation = activation;
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}

void forward_activation_layer(layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_activation_layer(layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void forward_activation_layer_gpu(layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_activation_layer_gpu(layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif
