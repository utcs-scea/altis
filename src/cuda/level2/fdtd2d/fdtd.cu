#include "fdtd.h"

using namespace std;

void RunBenchmark(ResultDATAbase &result DB, OptionParser &op) {
    cout << "Running FDTD" << endl;
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
}
