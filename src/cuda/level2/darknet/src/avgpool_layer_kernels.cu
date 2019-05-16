#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "avgpool_layer.h"
#include "cuda.h"
}

__global__ void forward_avgpool_layer_kernel(int n, int w, int h, int c, float *input, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    output[out_index] = 0;
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        output[out_index] += input[in_index];
    }
    output[out_index] /= w*h;
}

__global__ void backward_avgpool_layer_kernel(int n, int w, int h, int c, float *in_delta, float *out_delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        in_delta[in_index] += out_delta[out_index] / (w*h);
    }
}

extern "C" void forward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
    //size_t n = layer.c*layer.batch;

    //forward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.input_gpu, layer.output_gpu);

    cudaProfilerStart();
    float one = 1;
    cudnnStatus_t stat = cudnnPoolingForward(
            cudnn_handle(), layer.poolingDesc, &one, layer.poolingInputTensorDesc,
            net.input_gpu, &one, layer.poolingOutputTensorDesc, layer.output_gpu);
    assert(stat == CUDNN_STATUS_SUCCESS);

    check_error(cudaPeekAtLastError());
    cudaProfilerStop();
}

extern "C" void backward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
    //size_t n = layer.c*layer.batch;

    //backward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.delta_gpu, layer.delta_gpu);
    float one = 1;
    cudnnStatus_t stat = cudnnPoolingBackward(cudnn_handle(), layer.poolingDesc, &one,
            layer.poolingOutputTensorDesc,
            layer.output_gpu, layer.poolingOutputTensorDesc, layer.output_gpu,
            layer.poolingInputTensorDesc, net.input_gpu, &one, layer.poolingInputTensorDesc,
            net.delta_gpu);
    assert(stat == CUDNN_STATUS_SUCCESS);
    check_error(cudaPeekAtLastError());
}

