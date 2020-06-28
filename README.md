# Altis

This benchmark assumes availability of CUDA and CUDNN. To install CUDA; follow the instruction on [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). To install CUDNN, follow the instruction from [Deep Learning SDK Documentation](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html). This benchmark is tested on a machine with Ubuntu 18.04.4 LTS, gcc 7.4, and CUDA 10.0. The CUDNN version we used is 7.6.5.

## To Use:
1. ```git clone https://github.com/utcs-scea/altis.git```
2. ```cd altis```
3. ```bash configure --prefix=$ALTIS_ROOT```
4. ```make```

Note: If the configure script fails to execute due to dependency issues in automake toolchain, run the following command before executing ```configure```:
* ```libtoolize --force```
* ```aclocal```
* ```autoheader```
* ```automake --force-missing --add-missing```
* ```autoconf```

<!--
## To Run Suite:
``` python driver.py [options]```
```
Options:
  -h, --help                    show help message and exit
  -p, --prefix=PREFIX           location of Altis root, defaults to current working directory
  -e, --exec_prefix=EXEC_PREFIX location of executables
  -d, --device=DEVICE           device to run the benchmarks on
  -s, --size=SIZE               problem size
  -b, --benchmark=BENCHMARKS    comma-separated list of benchmarks to run, or 'all' to run entire suite, defaults to 'all'
  -v, --verbose                 enable verbose output
```
Note: Results are written to ```$ALTIS_ROOT/results/$BENCHMARK```
-->

## To Run a Single Benchmark:
1. ```cd src/cuda/$BENCHMARK_LEVEL/$BENCHMARK```
2. ``` ./$BENCHMARK [options]```
```
General Options: 
    -c, --configFile             specify configuration file
    -d, --device                 specify device to run on
    -i, --inputFile              path of input file
    -o, --outputFile             path of output file
    -m, --metricsFile            path of file to write metrics to
    -n, --passes                 specify number of passes
    -p, --properties             show properties for available platforms and devices (exits afterwards)
    -q, --quiet                  enable minimal output
    -s, --size                   specify problem size
    -v, --verbose                enable verbose output
```
Note: Run benchmark with --help to see full list of options available for that specific benchmark

## To Run a Benchmark with Custom Data:
1. ```python data/$BENCHMARK/datagen.py [options]```
2. Run benchmark with ```-i $DATA_FILEPATH```

Note: Not all benchmarks have a datagen
