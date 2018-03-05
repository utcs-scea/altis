#!/bin/bash

#./datagen 100
#./datagen 300
#./datagen 1000
#./datagen 3000
#./datagen 10000
#./datagen 30000
#./datagen 100000
#./datagen 300000
#./datagen 1000000
#./datagen 3000000
#./datagen 10000000
#./datagen 100 -f
#./datagen 300 -f
#./datagen 1000 -f
#./datagen 3000 -f
#./datagen 10000 -f
#./datagen 30000 -f
#./datagen 100000 -f
#./datagen 300000 -f
#./datagen 1000000 -f
#./datagen 3000000 -f
#./datagen 10000000 -f

python3 datagen.py -n 1024
python3 datagen.py -n 2048
python3 datagen.py -n 4096
python3 datagen.py -n 8192
python3 datagen.py -n 65536
python3 datagen.py -n 131072
python3 datagen.py -n 261444
python3 datagen.py -n 524288
python3 datagen.py -n 1048576
python3 datagen.py -n 2097152
python3 datagen.py -n 4194304
python3 datagen.py -n 8388608
python3 datagen.py -n 1024 -f
python3 datagen.py -n 2048 -f
python3 datagen.py -n 4096 -f
python3 datagen.py -n 8192 -f
python3 datagen.py -n 65536 -f
python3 datagen.py -n 131072 -f
python3 datagen.py -n 261444 -f
python3 datagen.py -n 524288 -f
python3 datagen.py -n 1048576 -f
python3 datagen.py -n 2097152 -f
python3 datagen.py -n 4194304 -f
python3 datagen.py -n 8388608 -f

