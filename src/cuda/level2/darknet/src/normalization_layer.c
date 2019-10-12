#include "normalization_layer.h"
#include "blas.h"

#include <stdio.h>

void test_normalization_layer_forward(int batch, int width, int height, int channel,
                int size, float alpha, float beta, float kappa) {
    printf("----- normalization forward -----\n");
    layer l = make_normalization_layer(batch, width, height, channel, size, alpha,
            beta, kappa);
    network *net = make_network(1);
    net->input_gpu = cuda_make_array(NULL, l.batch*l.h*l.w*l.c);
    cudaProfilerStart();
    forward_normalization_layer_gpu(l, *net);
    cudaProfilerStop();
    free_layer(l);
    free_network(net);
    printf("---------------------------------\n\n");
}

void test_normalization_layer_backward(int batch, int width, int height, int channel,
                int size, float alpha, float beta, float kappa) {
    printf("----- normalization backward -----\n");
    layer l = make_normalization_layer(batch, width, height, channel, size, alpha,
            beta, kappa);
    network *net = make_network(1);
    net->input_gpu = cuda_make_array(NULL, l.batch*l.h*l.w*l.c);
    cudaProfilerStart();
    backward_normalization_layer_gpu(l, *net);
    cudaProfilerStop();
    free_layer(l);
    free_network(net);
    printf("----------------------------------\n\n");
}

layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa)
{
    fprintf(stderr, "LRU Layer: %d x %d x %d image, %d size\n", w,h,c,size);
    layer layer = {0};
    layer.type = NORMALIZATION;
    layer.batch = batch;
    layer.h = layer.out_h = h;
    layer.w = layer.out_w = w;
    layer.c = layer.out_c = c;
    layer.kappa = kappa;
    layer.size = size;
    layer.alpha = alpha;
    layer.beta = beta;
    layer.inputs = w*h*c;
    layer.outputs = layer.inputs;

    #ifdef GPU
    layer.forward_gpu = forward_normalization_layer_gpu;
    layer.backward_gpu = backward_normalization_layer_gpu;

    layer.output_gpu =  cuda_make_array(0, h * w * c * batch);
    layer.delta_gpu =   cuda_make_array(0, h * w * c * batch);
    layer.squared_gpu = cuda_make_array(0, h * w * c * batch);
    layer.norms_gpu =   cuda_make_array(0, h * w * c * batch);
    layer.dy = cuda_make_array(0, h * w * c * batch);
#ifdef CUDNN
    
    cudnnCreateLRNDescriptor(&layer.LRNDesc);
    // all default values
    cudnnSetLRNDescriptor(layer.LRNDesc, 5, 0.0001, 0.75, 2.0);
    layer.lrnMode = CUDNN_LRN_CROSS_CHANNEL_DIM1;

    cudnnCreateTensorDescriptor(&layer.LRNInputTensorDesc);
    cudnnSetTensor4dDescriptor(layer.LRNInputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            layer.batch, layer.c, layer.h, layer.w);

#endif
    #endif
    return layer;
}

void resize_normalization_layer(layer *layer, int w, int h)
{
    int c = layer->c;
    int batch = layer->batch;
    layer->h = h;
    layer->w = w;
    layer->out_h = h;
    layer->out_w = w;
    layer->inputs = w*h*c;
    layer->outputs = layer->inputs;
#ifdef GPU
    cuda_free(layer->output_gpu);
    cuda_free(layer->delta_gpu); 
    cuda_free(layer->squared_gpu); 
    cuda_free(layer->norms_gpu);   
    layer->output_gpu =  cuda_make_array(0, h * w * c * batch);
    layer->delta_gpu =   cuda_make_array(0, h * w * c * batch);
    layer->squared_gpu = cuda_make_array(0, h * w * c * batch);
    layer->norms_gpu =   cuda_make_array(0, h * w * c * batch);
#endif
}

#ifdef GPU
void forward_normalization_layer_gpu(const layer layer, network net)
{
    /*
    int k,b;
    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
    scal_gpu(w*h*c*layer.batch, 0, layer.squared_gpu, 1);

    for(b = 0; b < layer.batch; ++b){
        float *squared = layer.squared_gpu + w*h*c*b;
        float *norms   = layer.norms_gpu + w*h*c*b;
        float *input   = net.input_gpu + w*h*c*b;
        pow_gpu(w*h*c, 2, input, 1, squared, 1);

        const_gpu(w*h, layer.kappa, norms, 1);
        for(k = 0; k < layer.size/2; ++k){
            axpy_gpu(w*h, layer.alpha, squared + w*h*k, 1, norms, 1);
        }

        for(k = 1; k < layer.c; ++k){
            copy_gpu(w*h, norms + w*h*(k-1), 1, norms + w*h*k, 1);
            int prev = k - ((layer.size-1)/2) - 1;
            int next = k + (layer.size/2);
            if(prev >= 0)      axpy_gpu(w*h, -layer.alpha, squared + w*h*prev, 1, norms + w*h*k, 1);
            if(next < layer.c) axpy_gpu(w*h,  layer.alpha, squared + w*h*next, 1, norms + w*h*k, 1);
        }
    }
    pow_gpu(w*h*c*layer.batch, -layer.beta, layer.norms_gpu, 1, layer.output_gpu, 1);
    mul_gpu(w*h*c*layer.batch, net.input_gpu, 1, layer.output_gpu, 1);
    */
    float alpha = 1;
    float beta = 0;
    cudnnLRNCrossChannelForward(cudnn_handle(), layer.LRNDesc, layer.lrnMode, &alpha,
            layer.LRNInputTensorDesc, net.input_gpu, &beta, layer.LRNInputTensorDesc,
            layer.output_gpu);
}

void backward_normalization_layer_gpu(const layer layer, network net)
{
    // TODO This is approximate ;-)

    /*
    int w = layer.w;
    int h = layer.h;
    int c = layer.c;
    pow_gpu(w*h*c*layer.batch, -layer.beta, layer.norms_gpu, 1, net.delta_gpu, 1);
    mul_gpu(w*h*c*layer.batch, layer.delta_gpu, 1, net.delta_gpu, 1);
    */

    float alpha = 1;
    float beta = 0;
    cudnnLRNCrossChannelBackward(cudnn_handle(), layer.LRNDesc, layer.lrnMode,
            &alpha, layer.LRNInputTensorDesc, net.input_gpu, layer.LRNInputTensorDesc,
            layer.dy, layer.LRNInputTensorDesc, layer.output_gpu, &beta, layer.LRNInputTensorDesc,
            layer.delta_gpu);
}
#endif
