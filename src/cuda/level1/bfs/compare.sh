#!/bin/bash
for i in 1k 2k 4k 8k 16k 32k 64k 128k 256k 512k 1M 2M 4M 8M 16M
    do
        ./rbfs inputs/graph$i.txt > res_rodinia
        ./bfs -i inputs/graph$i.txt --resultsfile r.txt --verbose -n 1 -o log > res_mirovia
        (diff res_rodinia res_mirovia) >> test
        (diff result.txt r.txt) >> test_res
    done
