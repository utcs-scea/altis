#include "ResultDatabase.h"
#include "OptionParser.h"

void addBenchmarkSpecOptions(OptionParser &op) {
}

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
}
#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

struct DepthwiseArgs {
  // Input layer dimensions
  int batch;
  int in_height;
  int in_width;
  int in_channel;
  int filter_height;
  int filter_width;
  int stride_height;
  int stride_width;
  int pad_height;
  int pad_width;

  // Output layer dimensions
  int out_height;
  int out_width;
  int out_channel;
};

namespace cuda {
template<typename DType, int kFilterHeight, int kFilterWidth>
__global__ void __launch_bounds__(1024, 2)
DepthwiseConv2dForwardKernel(const DType* input,
                             const DType* filter,
                             const DepthwiseArgs args,
                             int num_outputs,
                             DType* output) {
  const int in_channel = args.in_channel;
  const int in_height = args.in_height;
  const int in_width = args.in_width;
  const int filter_height = kFilterHeight > 0 ? kFilterHeight : args.filter_height;
  const int filter_width = kFilterWidth > 0 ? kFilterWidth : args.filter_width;
  const int stride_height = args.stride_height;
  const int stride_width = args.stride_width;
  const int pad_height = args.pad_height;
  const int pad_width = args.pad_width;
  const int out_channel = args.out_channel;
  const int out_height = args.out_height;
  const int out_width = args.out_width;

  CUDA_KERNEL_LOOP(thread_id, num_outputs) {
    // Compute the indexes of this thread in the output.
    //
    // We want coalesced reads so we make sure that each warp reads
    // a contiguous chunk of memory.
    //
    // THIS IS PROBABLY WRONG, we are not doing coalesced reads
    // into the input, because of the depth multiplier division...
    const int out_w = thread_id % out_width;
    const int out_h = (thread_id / out_width) % out_height;
    const int out_c = (thread_id / out_width / out_height) % out_channel;
    const int out_b = thread_id / out_width / out_height / out_channel;
    const int in_c = out_c;

    // Data is stored in the following format (let's assume we
    // flatten the height and width into one contiguous dimension
    // called "P".
    //
    // B1C1P1 B1C1P2 ..... B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 ..... B2C2P1 B2C2P2 ....
    //
    // Each row contains in_channel * in_height * in_width values
    // for each sample in the batch.
    //
    // We can further flatten it into:
    //
    // B1C1P1 B1C1P2 .....
    // B1C2P1 B1C2P2 ....
    // B2C1P1 B2C1P2 .....
    // B2C2P1 B2C2P2 ....
    //
    // where each row is a contiguous array of all of the spatial
    // pixels for a given batch and input depth.  The following
    // loop unrolls across the filter dimensions for a given thread,
    // indexing into the filter value and the corresponding input
    // patch.
    //
    // We can compute the index into the patch once right here.
    const int input_offset_temp = (out_b * in_channel + in_c) * (in_height * in_width);
    const int filter_offset_temp = in_c * filter_height * filter_width;

    // Finally, we can iterate over the spatial dimensions and perform the
    // convolution, writing into the output at the end.
    //
    // We perform an additional optimization, where we can determine
    // whether the patch fits within the image indices statically, and
    // avoid boundary checking within the loop.
    const int input_h_start = out_h * stride_height - pad_height;
    const int input_w_start = out_w * stride_width - pad_width;
    const int input_h_end = input_h_start + filter_height;
    const int input_w_end = input_w_start + filter_width;

    DType sum = 0;
    if (input_h_start >= 0 && input_w_start >= 0 &&
        input_h_end < in_height && input_w_end < in_width) {
      // Loop that doesn't need to check for boundary conditions.
      for (int f_h = 0; f_h < filter_height; ++f_h) {
        const int in_h = input_h_start + f_h;
        const int filter_offset_h = filter_width * f_h;
        for (int f_w = 0; f_w < filter_width; ++f_w) {
          const int in_w = input_w_start + f_w;
          const int input_offset = (input_offset_temp) + (in_h * in_width) + in_w;
          const int filter_offset = filter_offset_temp + filter_offset_h + f_w;
          sum += ldg(input + input_offset) * ldg(filter + filter_offset);
        }
      }
    } else {
      // Loop that needs to check for boundary conditions.
      for (int f_h = 0; f_h < filter_height; ++f_h) {
        const int in_h = input_h_start + f_h;
        const int filter_offset_h = filter_width * f_h;
        for (int f_w = 0; f_w < filter_width; ++f_w) {
          const int in_w = input_w_start + f_w;
          // TODO(vrv): the in_h check can be done outside of this loop;
          // benchmark both methods to determine the better decision.
          if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
            const int in_w = input_w_start + f_w;
            const int input_offset = input_offset_temp + (in_h * in_width) + in_w;
            const int filter_offset = filter_offset_temp + filter_offset_h + f_w;
            sum += ldg(input + input_offset) * ldg(filter + filter_offset);
          }
        }
      }
    }
    output[thread_id] = sum;
  }
}
}
