///-------------------------------------------------------------------------------------------------
// file:	kmeans-common.cu.h
//
// summary:	Declares the kmeans.cu class
///-------------------------------------------------------------------------------------------------

#ifndef __KMEANS_COMMON_CU_H__
#define __KMEANS_COMMON_CU_H__

#include "cudacommon.h"

////////////////////////////////////////////////////////////////////////////////
// Parameters
////////////////////////////////////////////////////////////////////////////////
static const int DEFAULTRANK = 16;        // known to match input file. see comment on ReadInput<>, default is 24
static const int DEFAULTSTEPS = 1000;     // number of steps to perform. Can also do convergence test.
static const int DEFAULTCENTERS = 64;  // number of clusters to find

static const int SHMEMSIZE = SHARED_MEM_PER_BLOCK;       // default shared mem size
static const int COUNTER_SHMEMSIZE = (256 * sizeof(int));
static const int ACCUM_SHMEMSIZE = (SHMEMSIZE-COUNTER_SHMEMSIZE);
static const int SHMEMCNTR_FLOATS = (COUNTER_SHMEMSIZE/sizeof(float));
static const int SHMEMACCUM_FLOATS = (ACCUM_SHMEMSIZE/sizeof(float));
static const int CONSTMEMSIZE = CONST_MEM;    // default constant memory size
static const int CONSTRSRVSIZE = SHMEMSIZE; // TODO check


////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
#define UINT_BITS 32
typedef unsigned int uint;
typedef unsigned char uchar;

////////////////////////////////////////////////////////////////////////////////
// GPU-specific common definitions
////////////////////////////////////////////////////////////////////////////////
#define THREADBLOCK_SIZE (4 * SHARED_MEMORY_BANKS)
// #define THREADBLOCK_SIZE 256
#define WARP_COUNT 6
#define UMUL(a, b) ( (a) * (b) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )
#define MAXBLOCKTHREADS MAX_THREADS_PER_BLOCK

////////////////////////////////////////////////////////////////////////////////
// GPU-specific common definitions
////////////////////////////////////////////////////////////////////////////////
//Data type used for input data fetches
typedef uint4 data_t;

////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute gridDim.x partial histograms
////////////////////////////////////////////////////////////////////////////////
//Count a byte into shared-memory storage
inline __device__ void addByte(uchar *s_ThreadBase, uint data){
    s_ThreadBase[UMUL(data, THREADBLOCK_SIZE)]++;
}

//Count four bytes of a word
inline __device__ void addWord(uchar *s_ThreadBase, uint data){
    //Only higher 6 bits of each byte matter, as this is a 64-bin histogram
    addByte(s_ThreadBase, (data >>  2) & 0x3FU);
    addByte(s_ThreadBase, (data >> 10) & 0x3FU);
    addByte(s_ThreadBase, (data >> 18) & 0x3FU);
    addByte(s_ThreadBase, (data >> 26) & 0x3FU);
}

//Round a / b to nearest higher integer value
inline uint iDivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Snap a to nearest lower multiple of b
inline uint iSnapDown(uint a, uint b){
    return a - a % b;
}

#ifdef _DEBUG
#define dassert(cond) {                                      \
    if((!(cond))&&(blockIdx.x*blockDim.x+threadIdx.x == 0)) {\
        printf("FAILED ASSERT: at %s, line %d\n",         \
               __FILE__,                                  \
               __LINE__);                                 \
    }}
#else
#define dassert(x)
#endif

#define INFORM(b,x) { if(b) {  shrLog(x); } }


#endif
