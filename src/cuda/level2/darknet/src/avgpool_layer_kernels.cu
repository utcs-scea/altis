#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "avgpool_layer.h"
#include "cuda.h"
}

extern "C" void forward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
    float one = 1;
    float zero = 0;
    cudnnStatus_t stat = cudnnPoolingForward(
            cudnn_handle(), layer.poolingDesc, &one, layer.poolingInputTensorDesc,
            net.input_gpu, &zero, layer.poolingOutputTensorDesc, layer.output_gpu);
    assert(stat == CUDNN_STATUS_SUCCESS);
}

extern "C" void backward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
    // As of cuDNN version 6.0, a deterministic algorithm is implemented for
    // max backwards pooling. This algorithm can be chosen via the pooling mode
    // enum of poolingDesc. The deterministic algorithm has been measured to be
    // up to 50% slower than the legacy max backwards pooling algorithm, or up
    // to 20% faster, depending upon the use case.
    float one = 1;
    float zero = 0;
    cudnnStatus_t stat = cudnnPoolingBackward(cudnn_handle(), layer.poolingDesc, &one,
            layer.poolingOutputTensorDesc,
            layer.output_gpu, layer.poolingOutputTensorDesc, layer.dy,
            layer.poolingInputTensorDesc, net.input_gpu, &zero, layer.poolingInputTensorDesc,
            layer.delta_gpu);
    assert(stat == CUDNN_STATUS_SUCCESS);
}

