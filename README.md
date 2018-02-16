# Mirovia

## To Install:
1. ```git clone https://github.com/sarahgrace/mirovia.git```
2. ```cd mirovia```
3. ```bash configure --prefix=$MIROVIA_ROOT```
4. ```make```
5. ```make install```

## To Run Suite:
``` python driver.py [options]```
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
2. ``` ./$BENCHMARK [options]```
```
General Options: 
    -c, --configFile             specify configuration file
    -d, --device                 specify device(s) to run on
    -h, --help                   print this usage
    -i, --inputFile              specify input file
    -p, --nopinned               disable usage of pinned (pagelocked) memory
    -o, --outputFile             specify output file
    -n, --passes                 specify number of passes
    -p, --properties             show properties for available platforms and devices
    -q, --quiet                  enable concise output
    -s, --size                   specify problem size
    -v, --verbose                enable verbose output
```
Note: Run benchmark with --help to see full list of options available for that specific benchmark

## To Run a Benchmark with Custom Data:
1. ```python data/$BENCHMARK/datagen.py [options]```
2. Run benchmark with ```--inputFile $DATA_FILEPATH```

Note: Not all benchmarks have a datagen
