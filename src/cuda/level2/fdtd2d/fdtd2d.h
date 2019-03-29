//#ifndef FDTD_H
//#define FDTD_H

// TODO: Maybe utilize half precision?
#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

void init_arrays(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz);
void run_fdtd_cpu(DATA_TYPE* _fict_, DATA_TYPE* ex, DATA_TYPE* ey, DATA_TYPE* hz);
void run_fdtd_cuda(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz);
void compare_results(DATA_TYPE *hz1, DATA_TYPE *hz2);


__global__ void kernel1(DATA_TYPE *_fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t);
__global__ void kernel2(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t);
__global__ void kernel3(DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t);

//#endif
