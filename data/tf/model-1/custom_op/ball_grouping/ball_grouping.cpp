#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cuda_runtime.h>

using namespace tensorflow;


/***************************/
/* CUDA Kernel Declaration */
/***************************/

void ball_grouping_launcher(
    int B, int N, int M, float radius, int nsample, 
    const float * global_points, const float * query_points,
    int * sample_index, int * unique_count
);


/*****************************/
/* Tensorflow Op Definitions */
/*****************************/

REGISTER_OP("BallGrouping")
    .Input("globals: float32")
    .Input("queries: float32")
    .Attr("num: int")
    .Attr("radius: float")
    .Output("idx: int32")
    .Output("unique_count: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        ::tensorflow::shape_inference::ShapeHandle dims2;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &dims2));
        int num;
        TF_RETURN_IF_ERROR(c->GetAttr("num", &num));
        ::tensorflow::shape_inference::ShapeHandle outshape_0 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), num});
        c->set_output(0, outshape_0);  // [B,M,K]
        ::tensorflow::shape_inference::ShapeHandle outshape_1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(1, outshape_1);  // [B,M]
        return Status::OK();
    });

class BallGroupingOp: public OpKernel {
  public:
    explicit BallGroupingOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("num", &num_));
        OP_REQUIRES(context, num_ > 0, errors::InvalidArgument("Expect positive num of samples!"));
        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
        OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("Expect positive maximum radius!"));
    }

    void Compute(OpKernelContext * context) override {
        // acquire & check inputs
        const Tensor& global_points = context->input(0);
        OP_REQUIRES(
            context, 
            global_points.dims()==3 && global_points.dim_size(2) == 3, 
            errors::InvalidArgument("Global points should have shape of (B,N,3)!")
        );

        const Tensor& query_points = context->input(1);
        OP_REQUIRES(
            context, 
            query_points.dims()==3 && query_points.dim_size(2) == 3, 
            errors::InvalidArgument("Query points should have shape of (B,N,3)!")
        );

        // create outputs
        const int B = global_points.dim_size(0);
        const int N = global_points.dim_size(1);
        const int M = query_points.dim_size(1);

        Tensor * idx_output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B,M,num_}, &idx_output));
        Tensor * count_output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{B,M}, &count_output));

        // launch cuda kernel
        const float * global_ptr = &(global_points.flat<float>()(0));
        const float * queries_ptr = &(query_points.flat<float>()(0));
        int * idx_ptr = &(idx_output->flat<int>()(0));
        int * count_ptr = &(count_output->flat<int>()(0));
        ball_grouping_launcher(B,N,M, radius_, num_, global_ptr, queries_ptr, idx_ptr, count_ptr);
    }

  private:
    int num_;
    float radius_;
};

REGISTER_KERNEL_BUILDER(Name("BallGrouping").Device(DEVICE_GPU), BallGroupingOp);
