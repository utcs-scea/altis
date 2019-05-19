#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

void test_maxpool_layer_forward(int batch, int height, int width, int chan,
                int size, int stride, int padding) {
    printf("----- maxpool forward -----\n");
    maxpool_layer l = make_maxpool_layer(batch, height, width, chan, size, stride, padding);
    network *net = make_network(1);
    net->input_gpu = cuda_make_array(NULL, l.batch*l.w*l.h*l.c);
    forward_maxpool_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("---------------------------\n\n");
}

void test_maxpool_layer_backward(int batch, int height, int width, int chan,
                int size, int stride, int padding) {
    printf("----- maxpool backward -----\n");
    maxpool_layer l = make_maxpool_layer(batch, height, width, chan, size, stride, padding);
    network *net = make_network(1);
    l.delta_gpu = cuda_make_array(NULL, l.w*l.h*l.c*l.batch);
    net->input_gpu = cuda_make_array(NULL, l.batch*l.w*l.h*l.c);
    l.dy = cuda_make_array(NULL, l.w*l.h*l.c*l.batch);
    backward_maxpool_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("----------------------------\n\n");
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
{
    maxpool_layer l = {0};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    #ifdef GPU
    l.forward_gpu = forward_maxpool_layer_gpu;
    l.backward_gpu = backward_maxpool_layer_gpu;
    l.indexes_gpu = cuda_make_int_array(0, output_size);
    l.output_gpu  = cuda_make_array(0, output_size);
    l.delta_gpu   = cuda_make_array(0, output_size);
    l.dy = cuda_make_array(0, output_size);

#ifdef CUDNN
    cudnnStatus_t stat = cudnnCreatePoolingDescriptor(&l.poolingDesc);
    assert(stat == CUDNN_STATUS_SUCCESS);
    // no padding for now
    stat = cudnnSetPooling2dDescriptor(l.poolingDesc, CUDNN_POOLING_MAX,
            CUDNN_NOT_PROPAGATE_NAN, l.h, l.w, l.pad, l.pad, l.stride, l.stride);
    assert(stat == CUDNN_STATUS_SUCCESS);

    stat = cudnnCreateTensorDescriptor(&l.poolingInputTensorDesc);
    assert(stat == CUDNN_STATUS_SUCCESS);

    stat = cudnnSetTensor4dDescriptor(l.poolingInputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            l.batch, l.c, l.h, l.w);
    assert(stat == CUDNN_STATUS_SUCCESS);

    stat = cudnnCreateTensorDescriptor(&l.poolingOutputTensorDesc);
    assert(stat == CUDNN_STATUS_SUCCESS);

    stat = cudnnSetTensor4dDescriptor(l.poolingOutputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            l.batch, l.out_c, l.out_h, l.out_w);
    assert(stat == CUDNN_STATUS_SUCCESS);


#endif
    #endif
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + l->pad - l->size)/l->stride + 1;
    l->out_h = (h + l->pad - l->size)/l->stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    #ifdef GPU
    cuda_free((float *)l->indexes_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->indexes_gpu = cuda_make_int_array(0, output_size);
    l->output_gpu  = cuda_make_array(0, output_size);
    l->delta_gpu   = cuda_make_array(0,  output_size);
    #endif
}

