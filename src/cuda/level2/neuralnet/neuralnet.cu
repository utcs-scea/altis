#include "ResultDatabase.h"
#include "OptionParser.h"
#include <dmlc/parameter.h>
#include <nnvm/op.h>
#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include "neuralnet/cuda_utils.h"


void addBenchmarkSpecOptions(OptionParser &op) {
}

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    DepthwiseArgs args;
    args.batch = 4;
    args.in_height = 1024;
    args.in_width = 1024;
    args.in_channel = 4;
    args.filter_height = 1024;
    args.filter_width = 1024;
    args.stride_height = 32;
    args.stride_width = 32;
    args.pad_height = 32;
    args.pad_width = 32;

    // Output layer dimensions
    args.out_height = 1024;
    args.out_width = 1024;
    args.out_channel = 4;

        DepthwiseConv2dForwardKernel<DType, 3, 3>
            <<<block_num, mshadow::cuda::kBaseThreadNum, 0, s>>>(data.dptr_,
                    weight.dptr_,
                    args,
                    num_output,
                    out.dptr_);
}

#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

namespace conv {
enum ConvolutionOpInputs {kData, kWeight, kBias};
enum ConvolutionOpOutputs {kOut};
enum ConvolutionOpResource {kTempSpace};
enum ConvolutionOpCudnnTune {kOff, kLimited, kFastest};
}

namespace tf {
namespace depthwise_conv {
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
} //cuda
} //depthwise_conv
} //tf

namespace op {
using namespace tf::depthwise_conv;
template<typename DType>
class DepthwiseConvolutionOp {
 public:
  void Init(const ConvolutionParam& param,
            const std::vector<TShape>& in_shape,
            const std::vector<TShape>& out_shape) {
    args_.batch = in_shape[conv::kData][0];
    args_.in_channel = in_shape[conv::kData][1];
    args_.in_height = in_shape[conv::kData][2];
    args_.in_width = in_shape[conv::kData][3];
    args_.filter_height = in_shape[conv::kWeight][2];
    args_.filter_width = in_shape[conv::kWeight][3];
    args_.stride_height = param.stride[0];
    args_.stride_width = param.stride[1];
    args_.pad_height = param.pad[0];
    args_.pad_width = param.pad[1];
    args_.out_channel = out_shape[conv::kOut][1];
    args_.out_height = out_shape[conv::kOut][2];
    args_.out_width = out_shape[conv::kOut][3];
    bias_term_ = !param.no_bias;
  }

  ~DepthwiseConvolutionOp() {}

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data);

  /*
  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad);
    */

 private:
  DepthwiseArgs args_;
  bool bias_term_;
};  // class DepthwiseConvolutionOp
} // op

template<typename DType>
void ConvolutionCompute(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  int dtype = inputs[conv::kData].type_flag_;

  if (param.num_filter == param.num_group &&
      param.layout.value() == mshadow::kNCHW &&
      param.num_filter == inputs[conv::kData].shape_[1] &&
      param.kernel.ndim() == 2 &&
      param.dilate == mshadow::Shape2(1, 1) &&
      dtype == mshadow::kFloat32) {
    std::vector<TShape> in_shape(inputs.size());
    std::vector<TShape> out_shape(1, outputs[0].shape_);
    for (size_t i = 0; i < in_shape.size(); i++)
      in_shape[i] = inputs[i].shape_;
    DepthwiseConvolutionOp<float> op;
    op.Init(param, in_shape, out_shape);
    op.Forward(ctx, inputs, req, outputs);
  }
}

template<typename DType>
void DepthwiseConv2dForwardGpu(mshadow::Stream<gpu> *stream,
        const DepthwiseArgs& args,
        const std::vector<TBlob> &in_data,
        const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    using namespace tf::depthwise_conv;
    using namespace tf::depthwise_conv::cuda;
    Tensor<gpu, 4, DType> data = in_data[conv::kData].get<gpu, 4, DType>(stream);
    Tensor<gpu, 4, DType> weight = in_data[conv::kWeight].get<gpu, 4, DType>(stream);
    Tensor<gpu, 4, DType> out = out_data[conv::kOut].get<gpu, 4, DType>(stream);

    int num_output = out_data[conv::kOut].shape_.Size();
    int block_num = std::min(num_output/mshadow::cuda::kBaseThreadNum + 1,
            mshadow::cuda::kMaxGridNum);
    auto s = mshadow::Stream<gpu>::GetStream(stream);
    if (args.filter_height == 3 && args.filter_width == 3) {
        DepthwiseConv2dForwardKernel<DType, 3, 3>
            <<<block_num, mshadow::cuda::kBaseThreadNum, 0, s>>>(data.dptr_,
                    weight.dptr_,
                    args,
                    num_output,
                    out.dptr_);
    }
}

template<typename DType>
void DepthwiseConvolutionOp<DType>::Forward(const OpContext &ctx,
                                            const std::vector<TBlob> &in_data,
                                            const std::vector<OpReqType> &req,
                                            const std::vector<TBlob> &out_data) {
  using namespace mshadow;
  using namespace mshadow::expr;
  auto stream = ctx.get_stream<gpu>();
  CHECK_EQ(req[conv::kOut], kWriteTo);
  // output forward
  depthwise_conv::DepthwiseConv2dForwardGpu<DType>(stream, args_, in_data, out_data);
}



