#include "dropout_layer.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>

void test_dropout_layer_forward(int batch, int input_size, float prob) {
    printf("----- testing dropout forward ------\n");
    dropout_layer l = make_dropout_layer(batch, input_size, prob);
    network *net = make_network(1);
    net->input_gpu = cuda_make_array(0, l.inputs*l.batch);
    net->train = 1;
    forward_dropout_layer_gpu(l, *net);
    //backward_dropout_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("------------------------------------\n\n");
}

void test_dropout_layer_backward(int batch, int input_size, float prob) {
    printf("----- testing dropout backward ------\n");
    dropout_layer l = make_dropout_layer(batch, input_size, prob);
    network *net = make_network(1);
    net->input_gpu = cuda_make_array(0, l.inputs*l.batch);
    net->delta_gpu = cuda_make_array(0, l.inputs*l.batch);
    backward_dropout_layer_gpu(l, *net);
    free_layer(l);
    free_network(net);
    printf("-------------------------------------\n\n");
}

dropout_layer make_dropout_layer(int batch, int inputs, float probability)
{
    dropout_layer l = {0};
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = calloc(inputs*batch, sizeof(float));
    l.scale = 1./(1.-probability);
    l.forward = forward_dropout_layer;
    l.backward = backward_dropout_layer;
    #ifdef GPU
    l.forward_gpu = forward_dropout_layer_gpu;
    l.backward_gpu = backward_dropout_layer_gpu;
    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
#ifdef CUDNN

    // TODO free
    cudnnStatus_t stat = cudnnCreateDropoutDescriptor(&l.dropoutDesc);
    assert(stat == CUDNN_STATUS_SUCCESS);
    
    stat = cudnnDropoutGetStatesSize(cudnn_handle(), &l.dropoutRandStatesSizeInBytes);
    assert(stat == CUDNN_STATUS_SUCCESS);
    
    l.dropoutRandStates = cuda_make_array(0, l.dropoutRandStatesSizeInBytes);
    // seed can be changed, but just for testing
    stat = cudnnSetDropoutDescriptor(l.dropoutDesc, cudnn_handle(), l.probability,
            l.dropoutRandStates, l.dropoutRandStatesSizeInBytes, 9);
    assert(stat == CUDNN_STATUS_SUCCESS);

    stat = cudnnCreateTensorDescriptor(&l.dropoutTensorDesc);
    assert(stat == CUDNN_STATUS_SUCCESS);
    const int dimA[4] = {l.batch, l.inputs, 1, 1};
    const int strideA[4] = {1, 1, 1, 1};
    stat = cudnnSetTensorNdDescriptor(l.dropoutTensorDesc, CUDNN_DATA_FLOAT, 4,
                    dimA, strideA);
    assert(stat == CUDNN_STATUS_SUCCESS);

    stat = cudnnDropoutGetReserveSpaceSize(l.dropoutTensorDesc, &l.dropoutReservedSize);
    assert(stat == CUDNN_STATUS_SUCCESS);
    //cuda_alloc_array(l.dropoutReservedSpace, l.dropoutReservedSize);
    l.dropoutReservedSpace = cuda_make_array(0, l.dropoutReservedSize);



#endif
    #endif
    fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
} 

void resize_dropout_layer(dropout_layer *l, int inputs)
{
    l->rand = realloc(l->rand, l->inputs*l->batch*sizeof(float));
    #ifdef GPU
    cuda_free(l->rand_gpu);

    l->rand_gpu = cuda_make_array(l->rand, inputs*l->batch);
    #endif
}

void forward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if (!net.train) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = rand_uniform(0, 1);
        l.rand[i] = r;
        if(r < l.probability) net.input[i] = 0;
        else net.input[i] *= l.scale;
    }
}

void backward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if(!net.delta) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = l.rand[i];
        if(r < l.probability) net.delta[i] = 0;
        else net.delta[i] *= l.scale;
    }
}

