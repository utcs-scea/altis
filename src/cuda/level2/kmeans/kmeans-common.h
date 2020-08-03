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
#define DEFAULTRANK 16                    // known to match input file. see comment on ReadInput<>, default is 24
#define DEFAULTSTEPS 1000                 // number of steps to perform. Can also do convergence test.
#define DEFAULTCENTERS 64                 // number of clusters to find

#define COUNTER_SHMEMSIZE (256 * sizeof(int))
#define ACCUM_SHMEMSIZE (SHARED_MEM_PER_BLOCK-COUNTER_SHMEMSIZE)
#define SHMEMCNTR_FLOATS (COUNTER_SHMEMSIZE/sizeof(float))
#define SHMEMACCUM_FLOATS (ACCUM_SHMEMSIZE/sizeof(float))
#define CONSTRSRVSIZE (CONST_MEM - 0x00150);    // reserved const mem


////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;
typedef unsigned char uchar;

////////////////////////////////////////////////////////////////////////////////
// GPU-specific common definitions
////////////////////////////////////////////////////////////////////////////////
// #define THREADBLOCK_SIZE (4 * SHARED_MEMORY_BANKS)
#define THREADBLOCK_SIZE MAX_THREADS_PER_BLOCK / 4
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

#define sharedAccumAvailable(R, C) ((R*C*sizeof(float) <= ACCUM_SHMEMSIZE) && (C*sizeof(int) <= COUNTER_SHMEMSIZE)) 

#define buildcoopkernel(ptr, kernel, R, C, nElemsPerThread, bRO, ROWMAJ, sharedAccum, sharedFinalize, accumulateGeneral)  \
    if (bRO == true && sharedAccum == true && sharedFinalize == true && accumulateGeneral == true)                \
        ptr = kernel<R,C,nElemsPerThread, true, ROWMAJ, true, true, true>;                                \
    else if (bRO == true && sharedAccum == true && sharedFinalize == true && accumulateGeneral == false)          \
        ptr = kernel<R,C,nElemsPerThread, true, ROWMAJ, true, true, false>;                               \
    else if (bRO == true && sharedAccum == true && sharedFinalize == false && accumulateGeneral == true)          \
        ptr = kernel<R,C,nElemsPerThread, true, ROWMAJ, true, false, true>;                               \
    else if (bRO == true && sharedAccum == true && sharedFinalize == false && accumulateGeneral == false)         \
        ptr = kernel<R,C,nElemsPerThread, true, ROWMAJ, true, false, false>;                              \
    else if (bRO == true && sharedAccum == false && sharedFinalize == true && accumulateGeneral == true)          \
        ptr = kernel<R,C,nElemsPerThread, true, ROWMAJ, false, true, true>;                               \
    else if (bRO == true && sharedAccum == false && sharedFinalize == true && accumulateGeneral == false)         \
        ptr = kernel<R,C,nElemsPerThread, true, ROWMAJ, false, true, false>;                              \
    else if (bRO == true && sharedAccum == false && sharedFinalize == false && accumulateGeneral == true)         \
        ptr = kernel<R,C,nElemsPerThread, true, ROWMAJ, false, false, true>;                              \
    else if (bRO == true && sharedAccum == false && sharedFinalize == false && accumulateGeneral == false)        \
        ptr = kernel<R,C,nElemsPerThread, true, ROWMAJ, false, false, false>;                             \
    else if (bRO == false && sharedAccum == true && sharedFinalize == true && accumulateGeneral == true)          \
        ptr = kernel<R,C,nElemsPerThread, false, ROWMAJ, true, true, true>;                               \
    else if (bRO == false && sharedAccum == true && sharedFinalize == true && accumulateGeneral == false)         \
        ptr = kernel<R,C,nElemsPerThread, false, ROWMAJ, true, true, false>;                              \
    else if (bRO == false && sharedAccum == true && sharedFinalize == false && accumulateGeneral == true)         \
        ptr = kernel<R,C,nElemsPerThread, false, ROWMAJ, true, false, true>;                              \
    else if (bRO == false && sharedAccum == true && sharedFinalize == false && accumulateGeneral == false)        \
        ptr = kernel<R,C,nElemsPerThread, false, ROWMAJ, true, false, false>;                             \
    else if (bRO == false && sharedAccum == false && sharedFinalize == true && accumulateGeneral == true)         \
        ptr = kernel<R,C,nElemsPerThread, false, ROWMAJ, false, true, true>;                              \
    else if (bRO == false && sharedAccum == false && sharedFinalize == true && accumulateGeneral == false)        \
        ptr = kernel<R,C,nElemsPerThread, false, ROWMAJ, false, true, false>;                             \
    else if (bRO == false && sharedAccum == false && sharedFinalize == false && accumulateGeneral == true)        \
        ptr = kernel<R,C,nElemsPerThread, false, ROWMAJ, false, false, true>;                             \
    else if (bRO == false && sharedAccum == false && sharedFinalize == false && accumulateGeneral == false)       \
        ptr = kernel<R,C,nElemsPerThread, false, ROWMAJ, false, false, false>;                            \
    else                                                                                                          \
        ptr = NULL;                                                                                               \

#endif
