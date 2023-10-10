#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "nanoflann.hpp"
#include "omp.h"

using namespace tensorflow;


/************************/
/* KNN Func Definitions */
/************************/

void kdtree_knn(
    const float* global_points, const size_t global_num, 
    const float* query_points, const size_t query_num, 
    const int k, const int dims,
    int * output_indice, float * output_dist
);

void batch_kdtree_knn(
    const float* global_points, const size_t global_num, 
    const float* query_points, const size_t query_num, 
    const int batch_size, const int k, const int dims,
    int * output_indice, float * output_dist
);

void batch_kdtree_knn_omp(
    const float* global_points, const size_t global_num, 
    const float* query_points, const size_t query_num, 
    const int batch_size, const int k, const int dims,
    int * output_indice, float * output_dist
);


/*****************************/
/* Tensorflow Op Definitions */
/*****************************/

REGISTER_OP("KnnGroupingANN")
    .Input("global_points: float32")
    .Input("query_points: float32")
    .Attr("K: int")
    .Attr("omp: bool = false")
    .Output("idx: int32")
    .Output("dist: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        ::tensorflow::shape_inference::ShapeHandle dims1, dims2;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims1));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &dims2));
        int K;
        TF_RETURN_IF_ERROR(c->GetAttr("K", &K));
        ::tensorflow::shape_inference::ShapeHandle output_shape = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), K});
        c->set_output(0, output_shape);
        c->set_output(1, output_shape);
        return Status::OK();
    });

class KnnGroupingANNOp: public OpKernel {
  public:
    explicit KnnGroupingANNOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("K", &K_));
        OP_REQUIRES(context, K_ > 0, errors::InvalidArgument("Expect positive K!"));
        OP_REQUIRES_OK(context, context->GetAttr("omp", &omp_));
    }

    void Compute(OpKernelContext* context) override {
        // acquire & check inputs
        const Tensor& global_points = context->input(0);
        OP_REQUIRES(
            context, 
            global_points.dims()==3, 
            errors::InvalidArgument("global_points should have shape of (B,N,3)!")
        );
        OP_REQUIRES(
            context, 
            K_ <= global_points.dim_size(1), 
            errors::InvalidArgument("K is bigger than the total number of points in global_points!")
        );

        const Tensor& query_points = context->input(1);
        OP_REQUIRES(
            context, 
            query_points.dims()==3, 
            errors::InvalidArgument("query_points should have shape of (B,N,D)!")
        );
        OP_REQUIRES(
            context,
            query_points.dim_size(2) == global_points.dim_size(2),
            errors::InvalidArgument("query_points should have same dims as global_points.")
        );

        // shape params
        const int B = global_points.dim_size(0);
        const int N = global_points.dim_size(1);
        const int M = query_points.dim_size(1);
        const int K = K_;
        const int D = global_points.dim_size(2);
        
        // create output tensor
        Tensor* nn_idx = NULL;
        Tensor* nn_dist = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B,M,K_}, &nn_idx));
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{B,M,K_}, &nn_dist));

        const float * g_pts = &(global_points.flat<float>()(0));
        const float * q_pts = &(query_points.flat<float>()(0));
        int * idx = &(nn_idx->flat<int>()(0));
        float * dist = &(nn_dist->flat<float>()(0));
        
        if (omp_) {
            batch_kdtree_knn_omp(g_pts, N, q_pts, M, B, K, D, idx, dist);
        }
        else {
            batch_kdtree_knn(g_pts, N, q_pts, M, B, K, D, idx, dist);
        }
        
    }
  private:
    int K_;
    bool omp_;
};

REGISTER_KERNEL_BUILDER(Name("KnnGroupingANN").Device(DEVICE_CPU), KnnGroupingANNOp);


/***********************/
/* KNN Implementations */
/***********************/

template<class TensorType, typename IndexType = size_t, int DIM = -1, class Distance = nanoflann::metric_L2>
struct KDTreeTensorAdaptor {
    typedef KDTreeTensorAdaptor<TensorType, IndexType, DIM, Distance> self_t;
    typedef typename Distance::template traits<TensorType, self_t>::distance_t metric_t;
    typedef nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM, IndexType> index_t;

    index_t *index; //! The kd-tree index for the user to call its methods as usual with any other FLANN index.
    const TensorType * data;
    const size_t npts;
    const size_t dims;

    KDTreeTensorAdaptor(
        const TensorType* data, const size_t npts, const size_t dims, const int leaf_max_size=10
    ) :data(data), npts(npts), dims(dims) {
        index = new index_t(static_cast<int>(dims), *this, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
        index->buildIndex();
    }

	~KDTreeTensorAdaptor() {
		delete index;
	}

    inline void query(const TensorType *query_point, const size_t k, IndexType *out_indices, TensorType * out_distances_sq) const {
        nanoflann::KNNResultSet<TensorType, IndexType> resultSet(k);
        resultSet.init(out_indices, out_distances_sq);
        index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
    }

    // Interface expected by KDTreeSingleIndexAdaptor
    const self_t &derived() const { return *this; }
    self_t &derived() { return *this; }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return this->npts;
    }

    // Returns the dim'th component of the idx'th point in the class
    inline TensorType kdtree_get_pt(const IndexType idx, size_t dim) const {
        return this->data[idx * this->dims + dim];
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    template <class BBOX> bool kdtree_get_bbox(BBOX & /*bb*/) const {
        return false;
    }
};

typedef KDTreeTensorAdaptor<float, int, -1, nanoflann::metric_L2> KDTree;

void kdtree_knn(
    const float* global_points, const size_t global_num, 
    const float* query_points, const size_t query_num, 
    const int k, const int dims,
    int * output_indice, float * output_dist
) {
    KDTree kdtree(global_points, global_num, dims, 10);
    for(int i = 0; i < query_num; i++) {
        const float * query = query_points + i * dims;
        int * out_indice = output_indice + i * k;
        float * out_dist = output_dist + i * k;
        kdtree.query(query, k, out_indice, out_dist);
    }
}

//! the global_num and query_num remains "num per batch" instead of total num
void batch_kdtree_knn(
    const float* global_points, const size_t global_num, 
    const float* query_points, const size_t query_num, 
    const int batch_size, const int k, const int dims,
    int * output_indice, float * output_dist
) {
    int global_batch_size = global_num * dims;
    int query_batch_size = query_num * dims;
    int output_batch_size = query_num * k;

    for(int b_idx = 0; b_idx < batch_size; b_idx++) {
        // locate pointers to current batch
        const float* batch_global = global_points + b_idx * global_batch_size;
        const float* batch_query = query_points + b_idx * query_batch_size;
        int * batch_output_indice = output_indice + b_idx * output_batch_size;
        float * batch_output_dist = output_dist + b_idx * output_batch_size;

        kdtree_knn(batch_global, global_num, batch_query, query_num, k, dims, batch_output_indice, batch_output_dist);
    }
}

//! the global_num and query_num remains "num per batch" instead of total num
void batch_kdtree_knn_omp(
    const float* global_points, const size_t global_num, 
    const float* query_points, const size_t query_num, 
    const int batch_size, const int k, const int dims,
    int * output_indice, float * output_dist
) {
    int global_batch_size = global_num * dims;
    int query_batch_size = query_num * dims;
    int output_batch_size = query_num * k;

# pragma omp parallel for
    for(int b_idx = 0; b_idx < batch_size; b_idx++) {
        const float* batch_global = global_points + b_idx * global_batch_size;
        const float* batch_query = query_points + b_idx * query_batch_size;
        int * batch_output_indice = output_indice + b_idx * output_batch_size;
        float * batch_output_dist = output_dist + b_idx * output_batch_size;

        kdtree_knn(batch_global, global_num, batch_query, query_num, k, dims, batch_output_indice, batch_output_dist);
    }
}
