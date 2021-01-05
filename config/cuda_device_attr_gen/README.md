# CUDA_Device_Attribute_generation
Automatically generate a C++ header file including Cuda device-specific parameters


## Getting Started

### Prerequisites

This project is tested on Linux, for Windows users, please refer to [cuda-samples](https://github.com/NVIDIA/cuda-samples) for additional instructions. Download and install the [CUDA Toolkit 11.0](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
For system requirements and installation instructions of cuda toolkit, please refer to the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/), and the [Windows Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

### Getting the tool

Using git clone the repository of CUDA_Device_Attribute_Generation using the command below.
```
git clone https://github.com/BDHU/CUDA_Device_Attribute_Generation.git
```
## How to Build

Simply run
```
make
```
It is only tested on Linux. To testing it on Windows, please refer to [cuda-samples](https://github.com/NVIDIA/cuda-samples) for guidances. The Makefile is modified to include header files in current directory instead of the original -I../../Common.

## How to Use

Simple run the executable using:
```
./deviceQuery [full path of the file to be generated] [device ID]
```

For example, you can run:
```
./deviceQuery ./ 0
```

This will generate a C/C++ header file named "cuda_device_attr.h" in the current working directory of the script. For multi_GPU users, simply change the device ID to generate headers for target GPUs in the system.
