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
#ifdef GPU
    l.forward_gpu = forward_activation_layer_gpu;
    l.backward_gpu = backward_activation_layer_gpu;

    l.output_gpu = cuda_make_array(0, inputs*batch);
    l.delta_gpu = cuda_make_array(0, inputs*batch);
    l.x_gpu = cuda_make_array(0, inputs*batch);

#ifdef CUDNN
    cudnnStatus_t stat = cudnnCreateActivationDescriptor(&l.activationDesc);
    assert(stat == CUDNN_STATUS_SUCCESS);
    switch (activation){
        case SIGMOID:
            l.activationMode = CUDNN_ACTIVATION_SIGMOID;
            break;
        case RELU: 
            l.activationMode = CUDNN_ACTIVATION_RELU;
            break;
        case TANH:
            l.activationMode = CUDNN_ACTIVATION_TANH;
            break;
        default:
            printf("This activation function is not supported or implemented yet, use normal activation instead.\n");
            exit(1);
    }
    // Not necessary here
    double coef = 0;
    stat = cudnnSetActivationDescriptor(l.activationDesc, l.activationMode,
                                CUDNN_NOT_PROPAGATE_NAN, coef);
    assert(stat == CUDNN_STATUS_SUCCESS);

    // set up activation tensor descriptor, wathc how chaning dimension will hit performance
    const int dimA[4] = {l.batch, l.inputs, 1, 1};
    const int strideA[4] = {1, 1, 1, 1};
    stat = cudnnCreateTensorDescriptor(&l.activationTensorDesc);
    assert(stat == CUDNN_STATUS_SUCCESS);
    /*
    stat = cudnnSetTensor4dDescriptor(l.activationTensorDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, l.batch, 1, 1, inputs);
            */
    stat = cudnnSetTensorNdDescriptor(l.activationTensorDesc, CUDNN_DATA_FLOAT, 4,
            dimA, strideA);
    assert(stat == CUDNN_STATUS_SUCCESS);
#endif
#endif
    l.activation = activation;
    fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    return l;
}

#ifdef GPU
void forward_activation_layer_gpu(layer l, network net)
{
#ifdef CUDNN
    float one = 1;
    float zero = 0;
    // in-place operation is allowed for this routine; i.e., xData and yData
    // pointers may be equal. However, this requires xDesc and yDesc descriptors
    // to be identical (particularly, the strides of the input and output must
    // match for in-place operation to be allowed).
    cudnnStatus_t stat = cudnnActivationForward(cudnn_handle(), l.activationDesc, &one,
            l.activationTensorDesc, net.input_gpu, &zero, l.activationTensorDesc,
            l.output_gpu);
    assert(stat == CUDNN_STATUS_SUCCESS);
#endif
}

void backward_activation_layer_gpu(layer l, network net)
{
    //gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    // requires gradient array gpu if applied in DNN
    float one = 1;
    float zero = 0;
#ifdef CUDNN
    cudnnStatus_t stat = cudnnActivationBackward(cudnn_handle(),
                            l.activationDesc,
                            &one,
                            l.activationTensorDesc,
                            l.output_gpu,
                            l.activationTensorDesc,
                            l.delta_gpu,
                            l.activationTensorDesc,
                            l.x_gpu,
                            &zero,
                            l.activationTensorDesc,
                            l.delta_gpu);
    assert(stat == CUDNN_STATUS_SUCCESS);
#endif
}
#endif
