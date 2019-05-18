#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "maxpool_layer.h"
#include "cuda.h"
}

extern "C" void forward_maxpool_layer_gpu(maxpool_layer layer, network net)
{
    float one = 1;
    cudnnStatus_t stat = cudnnPoolingForward(cudnn_handle(), layer.poolingDesc, &one,
            layer.poolingInputTensorDesc, net.input_gpu, &one, layer.poolingOutputTensorDesc,
            layer.output_gpu);
    assert(stat == CUDNN_STATUS_SUCCESS);
}

extern "C" void backward_maxpool_layer_gpu(maxpool_layer layer, network net)
{
    float one = 1;
    float zero = 0;
    cudnnStatus_t stat = cudnnPoolingBackward(cudnn_handle(), layer.poolingDesc, &one,
            layer.poolingOutputTensorDesc,
            layer.output_gpu, layer.poolingOutputTensorDesc, layer.dy,
            layer.poolingInputTensorDesc, net.input_gpu, &zero, layer.poolingInputTensorDesc,
            layer.delta_gpu);
    assert(stat == CUDNN_STATUS_SUCCESS);
}

