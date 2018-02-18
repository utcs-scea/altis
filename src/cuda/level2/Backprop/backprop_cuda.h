#ifndef BACKPROP_CUDA_H_
#define BACKPROP_CUDA_H_

__global__ void bpnn_layerforward_CUDA(float *input_cuda,
                                       float *output_hidden_cuda,
                                       float *input_hidden_cuda,
                                       float *hidden_partial_sum, int in,
                                       int hid);
__global__ void bpnn_adjust_weights_cuda(float *delta, int hid, float *ly,
                                         int in, float *w, float *oldw);

int setup(ResultDatabase &resultDB, OptionParser &op);
void backprop_face(int layer_size);
void bpnn_train_cuda(BPNN *net, float *eo, float *eh);

#endif // SORT_H_

