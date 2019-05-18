#include "connected_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test_connected_layer_forward(int batch, int input_size, int output_size,
            ACTIVATION actv, int batchnorm, int adam) 
{
    printf("Begin connected layer forward test...\n");
    layer l = make_connected_layer(batch, input_size, output_size, actv, 
            batchnorm, adam);
    network *net = make_network(1);
    net->input_gpu = cuda_make_array(l.weights, l.inputs*l.outputs);
    l.batch_normalize = 1;
    net->train = 1;
    forward_connected_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("\n");
}

void test_connected_layer_backward(int batch, int input_size, int output_size,
            ACTIVATION actv, int batchnorm, int adam) {
    printf("Begin connected layer backward test...\n");
    layer l = make_connected_layer(batch, input_size, output_size, actv,
            batchnorm, adam);
    network *net = make_network(1);
    net->input_gpu = cuda_make_array(l.weights, l.inputs*l.outputs);
    l.batch_normalize = 1;
    net->train = 1;
    backward_connected_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("\n");
}

layer make_connected_layer(int batch, int inputs, int outputs,
        ACTIVATION activation, int batch_normalize, int adam)
{
    int i;
    layer l = {0};
    l.learning_rate_scale = 1;
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch = batch;
    l.batch_normalize = batch_normalize;
    l.h = 1;
    l.w = 1;
    l.c = inputs;
    l.out_h = 1;
    l.out_w = 1;
    l.out_c = outputs;

#ifdef GPU
    l.forward_gpu = forward_connected_layer_gpu;
    l.backward_gpu = backward_connected_layer_gpu;
    l.update_gpu = update_connected_layer_gpu;

    l.weights_gpu = cuda_make_array(0, outputs*inputs);
    l.biases_gpu = cuda_make_array(0, outputs);

    l.weight_updates_gpu = cuda_make_array(0, outputs*inputs);
    l.bias_updates_gpu = cuda_make_array(0, outputs);

    l.output_gpu = cuda_make_array(0, outputs*batch);
    l.delta_gpu = cuda_make_array(0, outputs*batch);
    if (adam) {
        l.m_gpu =       cuda_make_array(0, inputs*outputs);
        l.v_gpu =       cuda_make_array(0, inputs*outputs);
        l.bias_m_gpu =  cuda_make_array(0, outputs);
        l.bias_v_gpu =  cuda_make_array(0, outputs);
        l.scale_m_gpu = cuda_make_array(0, outputs);
        l.scale_v_gpu = cuda_make_array(0, outputs);
    }

    if (batch_normalize) {
        l.mean_gpu = cuda_make_array(0, outputs);
        l.variance_gpu = cuda_make_array(0, outputs);

        l.rolling_mean_gpu = cuda_make_array(0, outputs);
        l.rolling_variance_gpu = cuda_make_array(0, outputs);

        l.mean_delta_gpu = cuda_make_array(0, outputs);
        l.variance_delta_gpu = cuda_make_array(0, outputs);

        l.scales_gpu = cuda_make_array(0, outputs);
        l.scale_updates_gpu = cuda_make_array(0, outputs);

        l.x_gpu = cuda_make_array(0, l.batch*outputs);
        l.x_norm_gpu = cuda_make_array(0, l.batch*outputs);
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
        cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 
        cudnnStatus_t stat = cudnnCreateActivationDescriptor(&l.activationDesc);
        assert(stat == CUDNN_STATUS_SUCCESS);

        switch (activation) {
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
                printf("This activation function not yet supported\n");
                exit(1);
        }

        double coef = 0;
        stat = cudnnSetActivationDescriptor(l.activationDesc, l.activationMode,
                CUDNN_NOT_PROPAGATE_NAN, coef);
        assert(stat == CUDNN_STATUS_SUCCESS);

        const int dimA[4] = {l.batch, l.outputs, 1, 1};
        const int strideA[4] = {1, 1, 1, 1};
        stat = cudnnCreateTensorDescriptor(&l.activationTensorDesc);
        assert(stat == CUDNN_STATUS_SUCCESS);

        stat = cudnnSetTensorNdDescriptor(l.activationTensorDesc, CUDNN_DATA_FLOAT, 4,
                dimA, strideA);
        assert(stat == CUDNN_STATUS_SUCCESS);
#endif
    }
#endif
    l.activation = activation;
    fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
    return l;
}

void statistics_connected_layer(layer l)
{
    if(l.batch_normalize){
        printf("Scales ");
        print_statistics(l.scales, l.outputs);
        /*
           printf("Rolling Mean ");
           print_statistics(l.rolling_mean, l.outputs);
           printf("Rolling Variance ");
           print_statistics(l.rolling_variance, l.outputs);
         */
    }
    printf("Biases ");
    print_statistics(l.biases, l.outputs);
    printf("Weights ");
    print_statistics(l.weights, l.outputs);
}

#ifdef GPU

void pull_connected_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void push_connected_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.outputs);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
    }
}

void update_connected_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;
    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.inputs*l.outputs, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.outputs, batch, a.t);
        }
    }else{
        axpy_gpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);

        if(l.batch_normalize){
            axpy_gpu(l.outputs, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.outputs, momentum, l.scale_updates_gpu, 1);
        }

        axpy_gpu(l.inputs*l.outputs, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.inputs*l.outputs, momentum, l.weight_updates_gpu, 1);
    }
}

void forward_connected_layer_gpu(layer l, network net)
{
    //fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float * a = net.input_gpu;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;
    gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

    /*
    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    }
    */
    cudnn_activate_array_gpu(l, net);
}

void backward_connected_layer_gpu(layer l, network net)
{
    //constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    //gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    // activation backward
    cudnn_gradient_array_gpu(l, net);
    /*
    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.outputs, 1);
    }
    */

    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float * a = l.delta_gpu;
    float * b = net.input_gpu;
    float * c = l.weight_updates_gpu;
    gemm_gpu(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta_gpu;
    b = l.weights_gpu;
    c = net.delta_gpu;

    if(c) gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
}
#endif
