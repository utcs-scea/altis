#include "avgpool_layer.h"
#include "cuda.h"
#include <stdio.h>

void test_avgpool_layer_forward(int batch, int width, int height, int chan) {
    printf("----- avgpool forward -----\n");
    avgpool_layer l = make_avgpool_layer(batch, width, height, chan);
    network *net = make_network(1);
    net->input_gpu = cuda_make_array(NULL, l.w*l.h*l.c*l.batch);
    forward_avgpool_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("-----------------------------");
    printf("\n\n");
}

void test_avgpool_layer_backward(int batch, int width, int height, int chan) {
    printf("----- avgpool backward -----\n");
    avgpool_layer l = make_avgpool_layer(batch, width, height, chan);
    network *net = make_network(1);
    l.delta_gpu = cuda_make_array(NULL, l.w*l.h*l.c*l.batch);
    net->input_gpu = cuda_make_array(NULL, l.w*l.h*l.c*l.batch);
    l.dy = cuda_make_array(NULL, l.w*l.h*l.c*l.batch);
    backward_avgpool_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("-----------------------------");
    printf("\n\n");
}

avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    avgpool_layer l = {0};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
#ifdef GPU
    l.forward_gpu = forward_avgpool_layer_gpu;
    l.backward_gpu = backward_avgpool_layer_gpu;
    l.output_gpu  = cuda_make_array(0, output_size);
    l.delta_gpu   = cuda_make_array(0, output_size);
#ifdef CUDNN

    cudnnStatus_t stat = cudnnCreatePoolingDescriptor(&l.poolingDesc);
    assert(stat == CUDNN_STATUS_SUCCESS);
    // no padding for now, may need to change parameter
    stat = cudnnSetPooling2dDescriptor(l.poolingDesc, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
            CUDNN_NOT_PROPAGATE_NAN, l.h, l.w, 0, 0, 1, 1);
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
    return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}

