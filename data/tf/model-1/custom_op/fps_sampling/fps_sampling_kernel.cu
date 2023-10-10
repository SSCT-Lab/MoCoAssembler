/*
  FPS主体分为两个步骤：
    1、计算各剩余点到已选集合的最短距离；
    2、从所有最短距离中选取距离最远的点加入已选集合；
  关键：
    1、各点到已选集合的最小距离仅需一次初始化；
    2、每当向已选集合中加入新的点，其余点到已选集合的最短距离只需通过min(old, new)来更新；
    3、分而治之思想，每个线程仅负责计算当前点到全局点云中的一部分点的最大值，之后再在各线程上同步；
*/
// 每个block负责(batch_size/gridDim.x)个batch，每个thread负责(num_points/blockDim.x)个点
__global__ void fps_kernel(
    int B,                              // batch_size
    int N,                              // num_points
    int M,                              // num_samples
    const float * __restrict__ data,    // [B,N,3]， 输入点云
    int * __restrict__ idxs,            // [B,M]，采样点坐标
    float * __restrict__ temp           // [gridDim.x,N]，临时数组，保存各点到已选取点的最小距离
){
    const int BlockSize = 512;  // gridDim.x
    __shared__ float dists[BlockSize];  // 各线程选取的最近点到已选集合的距离
    __shared__ int dists_i[BlockSize];  // 各线程选取的最近点在全局点集下的下标

    const int BufferSize = 3072;  // gridDim.x * 6
    __shared__ float buf[BufferSize*3];  // 用于保存缓存点的坐标

    // 依次处理当前线程负责的每个batch
    for (int i = blockIdx.x; i < B; i += gridDim.x){
        int old=0;  // 上一个被加入已选点集合的点的下标
        if (threadIdx.x==0) { idxs[i*M+0]=old; } // 如果当前节点是当前batch中的第一个节点，则使用该点作为已选点集合
            
        // 初始化当前线程所负责的各点到采样点集的最小距离
        for (int j = threadIdx.x; j < N; j += blockDim.x){
            temp[blockIdx.x*N+j]=1e38;  // {blockIdx.x*n}用于定位到当前block所对应的temp中的头部位置
        }

        // 从全局点云中将数据载入缓存(buffer)
        for (int j = threadIdx.x; j < min(BufferSize, N)*3; j += blockDim.x){
            buf[j]=data[i*N*3+j];
        }
        __syncthreads();

        // 循环m次以采样m个最远点
        for (int j = 1; j < M; j++){
            int besti=0; //距已选点集最远点的下标
            float best=-1; //最远的点到已选点集的距离

            // 提取最近被加入采样结果的点(即上一轮循环的结果)的坐标
            float x1=data[i*N*3+old*3+0];
            float y1=data[i*N*3+old*3+1];
            float z1=data[i*N*3+old*3+2];

            /* STEP-1: 从当前线程负责的所有点中选取距离采样集合距离最远的点 */
            for (int k = threadIdx.x; k < N; k += blockDim.x){  // 对线程负责的每一个点(索引为k)
                // 读取已知的最小距离
                float td=temp[blockIdx.x*N+k];

                //提取当前点的坐标
                float x2,y2,z2;
                if (k<BufferSize){  // 从缓存中读取
                    x2=buf[k*3+0];
                    y2=buf[k*3+1];
                    z2=buf[k*3+2];
                }
                else{  // 从输入中读取
                    x2=data[i*N*3+k*3+0];
                    y2=data[i*N*3+k*3+1];
                    z2=data[i*N*3+k*3+2];
                }

                // 根据最新加入采样集合的点更新当前点到采样集合的最小距离
                float d = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
                float d2 = min(d, td);
                if (d2!=td) { temp[blockIdx.x*N+k]=d2; }
                if (d2>best){ best=d2; besti=k; }
            }
            dists[threadIdx.x]=best;
            dists_i[threadIdx.x]=besti;

            /* STEP-2: 通过类似堆结构的上移操作在所有参与当前batch计算的线程的结果中选取最远点 */
            /*
            每个线程在dists中存放了其所处理的点中距离已选集合最远的点（即当前线程下的局部最远点），
            接下来需要从所有局部最远点中选出全局最远点作为最远点加入已选点集合.
            
            这里采用了二分法的思想，每轮迭代中区间被两两划分，每组的两个区间中，具有最大距离的点被
            放置在该组的最左侧，进行logn轮迭代后具有最大距离的点即被找到，并且该点位于[0]的位置。
            （可以视为最大堆的变种，每次向上一层进行ShiftUp操作。）

            注意由于线程数目固定，因此“堆”的大小也是固定的，同时需要注意，算法正常工作的前提是线程
            数能表示为2的指数，否则比较过程中会忽略超出2的指数的部分导致选取点可能不为局部最大。

            以8个线程为例（即从8个点中找出最大值）:
            u=0: 0 1 2 3 4 5 6 7  | 参与线程: 0, 1, 2, 3
            u=1: 0   2   4   6    | 参与线程: 0, 1
            u=2: 0       4        | 参与线程: 0
            u=3: 0                | 参与线程:
            此时距离最大点即位于0位置处。
            */
            for (int u = 0; (1<<u) < blockDim.x; u++){ //总共需要logn轮迭代
                __syncthreads();
                if (threadIdx.x < (blockDim.x>>(u+1))){ //只有当前迭代涉及到的线程需要执行
                    //if语句中的条件确保了在blockDim.x是2的指数时,有((threadIdx.x*2 + 1)<<u) < blockDim.x
                    int i1 = (threadIdx.x*2)<<u;  //当前线程的索引
                    int i2 = (threadIdx.x*2+1)<<u;  //i1节点的右邻居的索引
                    if (dists[i1]<dists[i2]){
                        dists[i1]=dists[i2];
                        dists_i[i1]=dists_i[i2];
                    }
                }
            }
            __syncthreads();
            old = dists_i[0];  // 更新最近加入采样点集合的点的索引
            if (threadIdx.x == 0) { idxs[i*M+j]=old; }  // 将新选取的采样点写入到输出(只需要一个线程进行写操作)
        }
    }
}

void fps_kernel_launcher(int B, int N, int M, const float * data, int * idxs, float * cache){
    fps_kernel<<<32, 512>>>(B, N, M, data, idxs, cache);
}