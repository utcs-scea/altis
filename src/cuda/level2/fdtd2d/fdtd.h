#ifndef FDTD_H
#define FDTD_H

#include <iostream>
#include <cstdlib>
#include <string>
#include <math.h>
#include <cassert>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"

// TODO may need to change threshold
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define DEFAULT_GPU 0

// TODO: porblem size, subject to change for benchmark purpose
#define tmax 500
#define NX 2048
#define NY 2048

// TODO: Maybe utilize half precision?
#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

void init_arrays(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz);
void run_fdtd_cuda(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz);
void compare_results(DATA_TYPE *hz1, DATA_TYPE *hz2);


__global__ void kernel1(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t);
__global__ void kernel2(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t);
__global__ void kernel3(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t);

#endif

