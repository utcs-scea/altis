#!/bin/bash

function die(){
    echo "$1"
    exit 1
}

# generate device-specific parameter header
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH/config/cuda_device_attr_gen

make 2>&1 >/dev/null || die "Failed to compile deviceQuery"
./deviceQuery ../../src/cuda/common/ 0 2>&1 >/dev/null || die "Failed to create cuda_device_attr.h"
make clean 2>&1 >/dev/null || die "Strange, failed to make clean for deviceQuery"
cd ..
#rm -rf cuda_device_attr_gen/ 2>&1 >/dev/null || die "Can't remove CUDA_Device_Attribute_Generation"

cd $SCRIPTPATH/src/cuda/common
if [[ ! -f "cuda_device_attr.h" ]]; then
    die "Didn't find cuda_device_attr.h, exiting..."
fi


cd $SCRIPTPATH
libtoolize --force
aclocal
autoheader
automake --force-missing --add-missing  # generate automake.in
autoconf    # generate configure
bash configure --prefix=$ALTIS_ROOT
make -j6
