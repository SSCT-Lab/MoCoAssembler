__global__ void ball_grouping_kernel(
    int N,                                      // num global points
    int M,                      				// num query points
    float radius,                             	// ball radius
    int nsample,                              	// samples per ball
    const float * __restrict__ global_points, 	// [B,N,3], global points batches
    const float * __restrict__ query_points,  	// [B,M,3], query points batches
    int * __restrict__ sample_index,          	// [B,M,nsamples], sample indexs
    int * __restrict__ unique_count           	// [B,M], num of unique samples
) {
    int batch_index = blockIdx.x;
    
    global_points += N*3*batch_index;
    query_points += M*3*batch_index;
    sample_index += M*nsample*batch_index;
    unique_count += M*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    // for each query points handled by current thread
    for (int i = index; i < M; i += stride) {
        float query_x = query_points[i*3 + 0];
        float query_y = query_points[i*3 + 1];
        float query_z = query_points[i*3 + 2];

        // loop over global points to acquire <nsample> sample
		int cnt = 0;
        for (int j = 0; j < N && cnt < nsample; j++) {
            float x = global_points[j*3 + 0];
            float y = global_points[j*3 + 1];
            float z = global_points[j*3 + 2];

            float dist2 = (x-query_x)*(x-query_x) + (y-query_y)*(y-query_y) + (z-query_z)*(z-query_z);
            float dist = max(sqrtf(dist2), 1e-20f);

            if (dist < radius) {
                // copy first sample "nsample" times incase there wasn't enough samples within "radius"
                if (cnt == 0) { for (int k = 0; k < nsample; k++) { sample_index[i*nsample+k] = j; } }
                sample_index[i*nsample+cnt] = j;
                cnt += 1;
            }
        }
        unique_count[i] = cnt;
    }
}

void ball_grouping_launcher(
    int B, int N, int M, float radius, int nsample, 
    const float * global_points, const float * query_points,
    int * sample_index, int * unique_count
) {
    ball_grouping_kernel<<<B, 512>>>(N, M, radius, nsample, global_points, query_points, sample_index, unique_count);
}
