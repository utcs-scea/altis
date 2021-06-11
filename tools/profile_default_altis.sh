#!/usr/bin/env bash
set -ue

LEVEL1_BENCH_PATH=/home/edwardhu/altis/src/cuda/level1/
LEVEL1_NUM_BENCH=0
LEVEL2_BENCH_PATH=/home/edwardhu/altis/src/cuda/level2/
LEVEL2_NUM_BENCH=8
ALL_LEVEL1_BENCHMARKS=(sort)
ALL_LEVEL2_BENCHMARKS=(cfd dwt2d kmeans lavamd mandelbrot nw particlefilter srad where)
ALL_DNN_BENCHMARKS=(activation avgpool batchnorm connected convolution dropout normalization rnn softmax)
DNN_BENCH_NUM=8
DNN_PATH=/home/edwardhu/altis/src/cuda/level2/darknet/

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
        ./run_big
    done
    
    # first on level2
    for i in $(seq 0 $LEVEL2_NUM_BENCH)
    do
        cd $LEVEL2_BENCH_PATH${ALL_LEVEL2_BENCHMARKS[$i]}
        ./run_big
    done

    # execute dnn kernel benchmark
    for i in $(seq 0 $DNN_BENCH_NUM)
    do
        cd $DNN_PATH${ALL_DNN_BENCHMARKS[$i]}_forward
        ./run_big

        cd $DNN_PATH${ALL_DNN_BENCHMARKS[$i]}_backward
        ./run_big
    done
}

profile_metrics_all_small_uvm () {
    # first on level1
    for i in $(seq 0 $LEVEL1_NUM_BENCH)
    do
        cd $LEVEL1_BENCH_PATH${ALL_LEVEL1_BENCHMARKS[$i]}
        ./run_small_uvm
    done
    
    # first on level2
    for i in $(seq 0 $LEVEL2_NUM_BENCH)
    do
        cd $LEVEL2_BENCH_PATH${ALL_LEVEL2_BENCHMARKS[$i]}
        ./run_small_uvm
    done

    # execute dnn kernel benchmark
    for i in $(seq 0 $DNN_BENCH_NUM)
    do
        cd $DNN_PATH${ALL_DNN_BENCHMARKS[$i]}_forward
        #./run_small_uvm

        cd $DNN_PATH${ALL_DNN_BENCHMARKS[$i]}_backward
        #./run_small_uvm
    done
}

profile_metrics_all_big_uvm () {
    # first on level1
    for i in $(seq 0 $LEVEL1_NUM_BENCH)
    do
        cd $LEVEL1_BENCH_PATH${ALL_LEVEL1_BENCHMARKS[$i]}
        ./run_big_uvm
    done
    
    # first on level2
    for i in $(seq 0 $LEVEL2_NUM_BENCH)
    do
        cd $LEVEL2_BENCH_PATH${ALL_LEVEL2_BENCHMARKS[$i]}
        ./run_big_uvm
    done

    # execute dnn kernel benchmark
    for i in $(seq 0 $DNN_BENCH_NUM)
    do
        cd $DNN_PATH${ALL_DNN_BENCHMARKS[$i]}_forward
        #./run_small_uvm

        cd $DNN_PATH${ALL_DNN_BENCHMARKS[$i]}_backward
        #./run_small_uvm
    done
}

#profile_metrics_all
#profile_metrics_all_small_uvm
profile_metrics_all_big_uvm
