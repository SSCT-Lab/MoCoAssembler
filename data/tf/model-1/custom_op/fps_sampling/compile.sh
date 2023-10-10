if !(test -v CUDA_HOME); then
    echo "Error, CUDA_HOME is not set!"
    exit -1
fi

if !(test -v TF_CFLAGS) || !(test -v TF_LFLAGS); then
    TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
    TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
fi

if !(test -v CUDA_CFLAGS) || !(test -v CUDA_LFLAGS); then
    CUDA_CFLAGS="-I${CUDA_HOME}/include"
    CUDA_LFLAGS="-L${CUDA_HOME}/lib64"
fi

nvcc -std=c++11 fps_sampling_kernel.cu -o fps_sampling_kernel.cu.o -c -O2 -x cu -Xcompiler -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]}
g++ -std=c++11 -shared fps_sampling.cpp fps_sampling_kernel.cu.o -o fps_sampling.so -fPIC  ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} ${CUDA_CFLAGS[@]} ${CUDA_LFLAGS[@]} -O2
