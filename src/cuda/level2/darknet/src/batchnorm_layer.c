#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include <stdio.h>

void test_batchnorm_layer_forward(int batch, int width, int height, int chan) {
    printf("----- batchnorm forward -----\n");
    layer l = make_batchnorm_layer(batch, width, height, chan);
    network *net = make_network(1);
    net->input_gpu = cuda_make_array(l.output, l.w*l.h*l.c*l.batch);
    // Can chose to test simple inference
    net->train = 1;
    forward_batchnorm_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("\n\n");
}

void test_batchnorm_layer_backward(int batch, int width, int height, int chan) {
    printf("----- batchnorm backward -----\n");
    layer l = make_batchnorm_layer(batch, width, height, chan);
    network *net = make_network(1);
    net->delta_gpu = cuda_make_array(l.delta, l.w*l.h*l.c*l.batch);
    // Can chose to test simple inference
    net->train = 1;
    backward_batchnorm_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("\n\n");
}

layer make_batchnorm_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w,h,c);
    layer l = {0};
    l.type = BATCHNORM;
    l.batch = batch;
    l.h = l.out_h = h;
    l.w = l.out_w = w;
    l.c = l.out_c = c;
    l.inputs = w*h*c;
    l.outputs = l.inputs;

#ifdef GPU
    l.forward_gpu = forward_batchnorm_layer_gpu;
    l.backward_gpu = backward_batchnorm_layer_gpu;

    l.output_gpu =  cuda_make_array(0, h * w * c * batch);
    l.delta_gpu =   cuda_make_array(0, h * w * c * batch);

    l.biases_gpu = cuda_make_array(0, c);
    l.bias_updates_gpu = cuda_make_array(0, c);

    l.scales_gpu = cuda_make_array(0, c);
    l.scale_updates_gpu = cuda_make_array(0, c);

    l.mean_gpu = cuda_make_array(0, c);
    l.variance_gpu = cuda_make_array(0, c);

    l.rolling_mean_gpu = cuda_make_array(0, c);
    l.rolling_variance_gpu = cuda_make_array(0, c);

    l.mean_delta_gpu = cuda_make_array(0, c);
    l.variance_delta_gpu = cuda_make_array(0, c);

    l.x_gpu = cuda_make_array(0, l.batch*l.outputs);
    l.x_norm_gpu = cuda_make_array(0, l.batch*l.outputs);
    #ifdef CUDNN
    cudnnCreateTensorDescriptor(&l.normTensorDesc);
    cudnnCreateTensorDescriptor(&l.dstTensorDesc);
    cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
    cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 
    #endif
#endif
    return l;
}

#ifdef GPU

void pull_batchnorm_layer(layer l)
{
    cuda_pull_array(l.scales_gpu, l.scales, l.c);
    cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}
void push_batchnorm_layer(layer l)
{
    cuda_push_array(l.scales_gpu, l.scales, l.c);
    cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}

void forward_batchnorm_layer_gpu(layer l, network net)
{
    //if(l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    //copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
    if (net.train) {
#ifdef CUDNN
        float one = 1;
        float zero = 0;
        cudnnBatchNormalizationForwardTraining(cudnn_handle(),
                CUDNN_BATCHNORM_SPATIAL,
                &one,
                &zero,
                l.dstTensorDesc,
                net.input_gpu,
                l.dstTensorDesc,
                l.output_gpu,
                l.normTensorDesc,
                l.scales_gpu,
                l.biases_gpu,
                .01,
                l.rolling_mean_gpu,
                l.rolling_variance_gpu,
                .00001,
                l.mean_gpu,
                l.variance_gpu);
#endif
    } else {
        normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
    }

}

void backward_batchnorm_layer_gpu(layer l, network net)
{
    if(!net.train){
        l.mean_gpu = l.rolling_mean_gpu;
        l.variance_gpu = l.rolling_variance_gpu;
    }
#ifdef CUDNN
    float one = 1;
    float zero = 0;
    cudnnBatchNormalizationBackward(cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one,
            &zero,
            &one,
            &one,
            l.dstTensorDesc,
            l.x_gpu,
            l.dstTensorDesc,
            l.delta_gpu,
            l.dstTensorDesc,
            l.x_norm_gpu,
            l.normTensorDesc,
            l.scales_gpu,
            l.scale_updates_gpu,
            l.bias_updates_gpu,
            .00001,
            l.mean_gpu,
            l.variance_gpu);
    copy_gpu(l.outputs*l.batch, l.x_norm_gpu, 1, l.delta_gpu, 1);
#else
    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h);
    backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates_gpu);

    scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);

    fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta_gpu);
    fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta_gpu);
    normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
#endif
    if(l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif
