#!/bin/bash

DEVICE_ID=0

helpFunction()
{
   echo ""
   echo "Usage: $0 -d <device-id>"
   echo -e "\t-d Specify target CUDA device ID (default to 0)"
   exit 1 # Exit script after printing help
}

while getopts "d:" opt
do
   case "$opt" in
      d ) DEVICE_ID="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

function die(){
    echo "$1"
    exit 1
}

# generate device-specific parameter header
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH/config/cuda_device_attr_gen

make 2>&1 >/dev/null || die "Failed to compile deviceQuery"
./deviceQuery ../../src/cuda/common/ $DEVICE_ID 2>&1 >/dev/null || die "Failed to create cuda_device_attr.h"
make clean 2>&1 >/dev/null || die "Strange, failed to make clean for deviceQuery"
cd ..
#rm -rf cuda_device_attr_gen/ 2>&1 >/dev/null || die "Can't remove CUDA_Device_Attribute_Generation"

cd $SCRIPTPATH/src/cuda/common
if [[ ! -f "cuda_device_attr.h" ]]; then
    die "Didn't find cuda_device_attr.h, exiting..."
fi


cd $SCRIPTPATH
mkdir build
cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=$($SCRIPTPATH/config/get_cuda_sm.sh) ..
make -j`nproc`
