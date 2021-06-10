#!/bin/sh
set -ue

./gemm -s 5 --passes 30 --uvm
