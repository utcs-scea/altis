#!/bin/bash
for i in 1024 2048 4096 8192 65536 131072 261444 524288 1048576 2097152 4194304 8388608
    do
        for j in 10 20 30 40 50 60 70 80 90 100
            do
                echo $i $j
                ./needle $i $j > res_rodinia
                ./nw --dimensions $i --penalty $j --resultsfile mirovia.txt -o log -n 1 > res_mirovia
                echo $i $j >> test
                (diff res_rodinia res_mirovia) >> test
                (diff rodinia.txt mirovia.txt) >> test_res
            done
    done

