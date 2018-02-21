#!/bin/bash
cd src/cuda
cd level0
for i in busspeedreadback busspeeddownload devicememory maxflops
do
    cd $i
    rm *.i
    rm *.ii
    rm *.ptx
    rm *.cubin
    rm *cudafe*
    rm *fatbin*
    rm *module_id*
    cd ..
done
cd ..
cd level1
for i in gemm sort spmv stencil2d
do
    cd $i
    rm *.i
    rm *.ii
    rm *.ptx
    rm *.cubin
    rm *cudafe*
    rm *fatbin*
    rm *module_id*
    cd ..
done
cd ..
cd level2
for i in backprop lavaMD
do
    cd $i
    rm *.i
    rm *.ii
    rm *.ptx
    rm *.cubin
    rm *cudafe*
    rm *fatbin*
    rm *module_id*
    cd ..
done
