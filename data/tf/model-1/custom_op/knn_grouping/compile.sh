if !(test -v TF_CFLAGS) || !(test -v TF_LFLAGS); then
    TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
    TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
fi


g++ -std=c++11 -shared knn_grouping.cpp -o knn_grouping.so -fopenmp -fPIC  ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
