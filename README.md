# Mirovia

## To Install:
1. ```git clone https://github.com/sarahgrace/mirovia.git```
2. ```cd mirovia```
3. ```bash configure```
4. ```make```
5. ```make install```

## To Run Suite:
``` python driver.py```

```
Options:
  -h, --help                    show help message and exit
  -p, --prefix=PREFIX           location of Mirovia root, defaults to current working directory
  -d, --device=DEVICE           device to run the benchmarks on
  -s, --size=SIZE               problem size
  -b, --benchmark=BENCHMARK     comma-separated list of benchmarks to run, or 'all' to run entire suite, defaults to 'all'
```
Note: Results are written to ```$MIROVIA_ROOT/results/$BENCHMARK```

## To Run a Single Benchmark:
1. ```cd src/cuda/$BENCHMARK_LEVEL/$BENCHMARK```
2. ``` ./$BENCHMARK```

```
Options: 
    -h, --help                   show help message and exit
    -i, --infoDevices            show info for available platforms and devices and exit
    -d, --device                 specify device(s) to run on
    -s, --size                   specify problem size
    -n, --passes                 specify number of passes
    -f, --outfile                specify output file for results
    -p, --nopinned               disable usage of pinned (pagelocked) memory
    -q, --quiet                  write minimum necessary to standard output
    -v, --verbose                enable verbose output
```
Note: Results are written to standard out unless an output file is specified

