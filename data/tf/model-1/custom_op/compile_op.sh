if !(test -v CUDA_HOME); then
    echo "Error, CUDA_HOME is not set!"
    exit -1
fi

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

CUDA_CFLAGS="-I${CUDA_HOME}/include"
CUDA_LFLAGS="-L${CUDA_HOME}/lib64"

# switch
build_fps_sampling=true
build_knn_grouping=true
build_ball_grouping=true

# FPS operation
if $build_fps_sampling; then
    echo "building fps sampling module..."
    cd fps_sampling; source compile.sh; cd ..
    if [ $? -ne 0 ]; then echo -e '\e[0;31mBUILD FAILED!\e[0m'; exit -1; fi
    echo "Done."
fi

# KNN grouping
if $build_knn_grouping; then
    echo "building knn grouping module..."
    cd knn_grouping; source compile.sh; cd ..
    if [ $? -ne 0 ]; then echo -e '\e[0;31mBUILD FAILED!\e[0m'; exit -1; fi
    echo "Done."
fi

# ball grouping
if $build_ball_grouping; then
    echo "building ball grouping module..."
    cd ball_grouping; source compile.sh; cd ..
    if [ $? -ne 0 ]; then echo -e '\e[0;31mBUILD FAILED!\e[0m'; exit -1; fi
    echo "Done."
fi
