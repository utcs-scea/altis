#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "dropout_layer.h"
#include "cuda.h"
#include "utils.h"
}

__global__ void dropout_kernel(float *input, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

void forward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if (!net.train) return;
    /*
    int size = layer.inputs*layer.batch;
    cuda_random(layer.rand_gpu, size);
    dropout_kernel<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, size, layer.rand_gpu,
                                            layer.probability, layer.scale);
    */
    
#ifdef CUDNN
    cudnnStatus_t stat = cudnnDropoutForward(cudnn_handle(), layer.dropoutDesc,
            layer.dropoutTensorDesc, net.input_gpu,
            layer.dropoutTensorDesc, net.input_gpu,
            layer.dropoutReservedSpace, layer.dropoutReservedSize);
    assert(stat == CUDNN_STATUS_SUCCESS);
#endif


    check_error(cudaPeekAtLastError());
}

void backward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if(!net.delta_gpu) return;
    /*
    int size = layer.inputs*layer.batch;

    dropout_kernel<<<cuda_gridsize(size), BLOCK>>>(net.delta_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    check_error(cudaPeekAtLastError());
    */
#ifdef CUDNN
    cudnnStatus_t stat = cudnnDropoutBackward(cudnn_handle(), layer.dropoutDesc,
            layer.dropoutTensorDesc, net.delta_gpu,
            layer.dropoutTensorDesc, net.delta_gpu,
            layer.dropoutReservedSpace, layer.dropoutReservedSize);
    assert(stat == CUDNN_STATUS_SUCCESS);

#endif
}
