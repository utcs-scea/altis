#!/usr/bin/env bash
set -ue

LEVEL1_BENCH_PATH=/home/edwardhu/mirovia/src/cuda/level1/
#LEVEL1_NUM_BENCH=12
LEVEL1_NUM_BENCH=1
LEVEL2_BENCH_PATH=/home/edwardhu/mirovia/src/cuda/level2/
#LEVEL2_NUM_BENCH=10
LEVEL2_NUM_BENCH=1
#ALL_LEVEL1_BENCHMARKS=(bfs fft gemm md md5hash neuralnet reduction scan sort spmv stencil2d triad)
#ALL_LEVEL1_BENCHMARKS=(gemm pathfinder sort)
ALL_LEVEL1_BENCHMARKS=(gemm)
#ALL_LEVEL2_BENCHMARKS=(cfd dwt2d fdtd2d gups kmeans lavamd mandelbrot nw particlefilter srad where)
ALL_LEVEL2_BENCHMARKS=(nw )


profile_events_all () {
    for i in $(seq 0 $NUM_BENCH)
    do
        cd $BENCH_PATH${ALL_BENCHMARKS[$i]}
        sudo /usr/local/cuda-10.0/bin/nvprof --profile-child-processes -e all --csv --log-file "%p" ./profile
        #nvprof --profile-child-processes -e all --csv --log-file "%p" ./profile
        echo ${ALL_BENCHMARKS[$i]}
    done
}

profile_metrics_all () {
    # first on level1
    for i in $(seq 0 $LEVEL1_NUM_BENCH)
    do
        cd $LEVEL1_BENCH_PATH${ALL_LEVEL1_BENCHMARKS[$i]}
        #nvprof --profile-child-processes -e all --csv --log-file "%p" ./profile
        #./run_small
        ./run_big
    done
    
    # first on level2
    for i in $(seq 0 $LEVEL2_NUM_BENCH)
    do
        cd $LEVEL2_BENCH_PATH${ALL_LEVEL2_BENCHMARKS[$i]}
        #nvprof --profile-child-processes -e all --csv --log-file "%p" ./profile
        ./run_big
    done

    # execute dnn kernel benchmark

}

profile_metrics_all
