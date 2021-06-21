# Altis Benchmark Suite

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](LICENSE)
![CI](https://github.com/utcs-scea/altis/actions/workflows/action.yml/badge.svg)
[![DOI:10.1109/ISPASS48437.2020.00011](https://zenodo.org/badge/DOI/10.1109/ISPASS48437.2020.00011.svg)](https://doi.org/10.1109/ISPASS48437.2020.00011)


Altis is a benchmark suite to test the performance and other aspects of systems with Graphics Processing Units (GPUs), developed in [SCEA](https://github.com/utcs-scea) lab at University of Texas at Austin. Altis consists of a collection of GPU applications with differnt performance implications. Altis focuses primarily on [Compute Unified Device Architecture](https://developer.nvidia.com/cuda-toolkit) (CUDA) computing platform.

Documentaion regarding this project can be found at the [Wiki](https://github.com/utcs-scea/altis/wiki) page. The Wiki document contains information regarding Altis setup, installation, usage, and other information.

> We are refactoring Altis codebase for better usability and making it more developer-friendly. We made sure the benchmark still compile properly during refactoring so you can still use it. The refactoring involves changing how each benchmark application is used and adding more benchmarks.

## How to Get Altis

Simply execute the following command:

```bash
git clone https://github.com/utcs-scea/altis.git
```

## Setup

Altis relies on the avaialbility of CUDA and CMake (>= 3.8). Please refer to [Environment Setup](https://github.com/utcs-scea/altis/wiki/Environment-Setup) for how to set up Altis.

## Build:

After the environment is setup properly, go to the root directory of Altis, execute:

```bash
./setup.sh
```

For more information regarding building process, please refer to [Build](https://github.com/utcs-scea/altis/wiki/Build) for more information.


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

## Cite Us

Bibtex is shown below:  

@INPROCEEDINGS{9238617,  
  author={B. {Hu} and C. J. {Rossbach}},  
  booktitle={2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},  
  title={Altis: Modernizing GPGPU Benchmarks},  
  year={2020},  
  volume={},  
  number={},  
  pages={1-11},  
  doi={10.1109/ISPASS48437.2020.00011}}  

## Publication

B. Hu and C. J. Rossbach, "Altis: Modernizing GPGPU Benchmarks," 2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), Boston, MA, USA, 2020, pp. 1-11, doi: 10.1109/ISPASS48437.2020.00011.

## Contact

For any questions regarding this project, please send an email to [bodunhu@utexas.edu](mailto:bodunhu@utexas.edu) or [rossbach@cs.utexas.edu](mailto:rossbach@cs.utexas.edu)
