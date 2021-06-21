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
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/.."
echo $SCRIPTPATH

cd $SCRIPTPATH/src/cuda/common
if [[ ! -f "cuda_device_attr.h" ]]; then
    die "Didn't find cuda_device_attr.h, exiting..."
fi


cd $SCRIPTPATH
mkdir build
cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=60\;61\;70\;75\;80\;86 ..
make -j`nproc`
