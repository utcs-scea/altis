#!/bin/bash
for i in $(seq 100000 100000 1000000)
do
    ./backprop $i > res_rodinia
    ./Backprop --layerSize $i > res_mirovia
    (diff res_rodinia res_mirovia) >> diff
    echo "*******************" >> diff
done
