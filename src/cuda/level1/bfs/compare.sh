#!/bin/bash
for i in 1024 2048 4096 8192 65536 131072 261444 524288 1048576 2097152 4194304 8388608
    do
        ./rbfs inputs/bfs_$i > res_rodinia
        ./bfs -i inputs/bfs_$i --resultsfile r.txt --verbose -n 1 -o log > res_mirovia
        (diff res_rodinia res_mirovia) >> test
        (diff result.txt r.txt) >> test_res
    done
