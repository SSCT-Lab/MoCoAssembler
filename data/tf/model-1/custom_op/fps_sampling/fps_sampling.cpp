#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cuda_runtime.h>

using namespace tensorflow;


/***************************/
/* CUDA Kernel Declaration */
/***************************/

void fps_kernel_launcher(int B, int N, int M, const float * data, int * idxs, float * cache);


/*****************************/
/* Tensorflow Op Definitions */
/*****************************/

REGISTER_OP("FarthestPointSample")
    .Input("pc: float32")
    .Attr("num: int")
    .Output("idx: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        ::tensorflow::shape_inference::ShapeHandle dims1;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims1));
        int num;
        TF_RETURN_IF_ERROR(c->GetAttr("num", &num));
        ::tensorflow::shape_inference::ShapeHandle output_shape = c->MakeShape({c->Dim(dims1, 0), num});
        c->set_output(0, output_shape);
        return Status::OK();
    });

class FarthestPointSampleOp: public OpKernel {
  public:
    explicit FarthestPointSampleOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("num", &num_));
        OP_REQUIRES(context, num_ > 0, errors::InvalidArgument("Expect positive num of samples!"));
    }
    
    void Compute(OpKernelContext * context) override {
        // acquire & check inputs
        const Tensor& points = context->input(0);
        OP_REQUIRES(
            context, 
            points.dims()==3, 
            errors::InvalidArgument("Input points should have shape of (B,N,3)!")
        );
        OP_REQUIRES(
            context, 
            num_ <= points.dim_size(1), 
            errors::InvalidArgument("Sample num is larger than total number of points!")
        );

        // create outputs and cache
        const int B = points.dim_size(0);
        const int N = points.dim_size(1);

        Tensor * idx_output = nullptr;
        Tensor cache;
        OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{B,num_}, &idx_output));
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value, TensorShape{32,N}, &cache));

        // launch cuda kernel
        const float * points_ptr = &(points.flat<float>()(0));
        int * idx_ptr = &(idx_output->flat<int>()(0));
        float * cache_ptr = &(cache.flat<float>()(0));
        fps_kernel_launcher(B, N, num_, points_ptr, idx_ptr, cache_ptr);
    }

  private:
    int num_;
};

REGISTER_KERNEL_BUILDER(Name("FarthestPointSample").Device(DEVICE_GPU), FarthestPointSampleOp);
