#!/bin/bash

#./graphgen 1024 1k
#./graphgen 2048 2k
#./graphgen 4096 4k
#./graphgen 8192 8k
#./graphgen 16384 16k
#./graphgen 32768 32k
#./graphgen 65536 64k
#./graphgen 131072 128k
#./graphgen 261444 256k
#./graphgen 524288 512k
#./graphgen 1048576 1M
#./graphgen 2097152 2M
#./graphgen 4194304 4M
#./graphgen 8388608 8M
#./graphgen 16777216 16M

python3 datagen.py -n 128
python3 datagen.py -n 256
python3 datagen.py -n 512
python3 datagen.py -n 1024
python3 datagen.py -n 2048
python3 datagen.py -n 4096
python3 datagen.py -n 8192
python3 datagen.py -n 16384
python3 datagen.py -n 32768
python3 datagen.py -n 65536
python3 datagen.py -n 131072
python3 datagen.py -n 261444
python3 datagen.py -n 524288
python3 datagen.py -n 1048576
python3 datagen.py -n 2097152
python3 datagen.py -n 4194304
python3 datagen.py -n 8388608


