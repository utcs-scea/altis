#include "cudacommon.h"
#include <cuda_runtime.h>
#include "genericvector.h"
#include "kmeans-common.h"
#include "cudacommon.h"

#ifdef GRID_SYNC

#include <cooperative_groups.h>
using namespace cooperative_groups;

typedef struct {
    float * pP;
    float * pC;
    int * pCC;
    int * pCI;
    int nP;
} kmeans_params;

#endif

__constant__ float d_cnst_centers[CONST_MEM / sizeof(float) - 0x00100];

/*
 * Reset kernel impl
 */
template <int R, int C>
__global__ void resetExplicit(float * pC, int * pCC)  {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= C) return;
    // printf("resetExplicit: pCC[%d]=>%.3f/%d\n", idx, pC[idx*R], pCC[idx]);
	for(int i=0;i<R;i++) 
		pC[idx*R+i] = 0.0f;
    pCC[idx] = 0;
}

template <int R, int C>
__global__ void resetExplicitColumnMajor(float * pC, int * pCC)  {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= C) return;
    // printf("resetExplicitColumnMajor: pCC[%d]=>%.3f/%d\n", idx, pC[idx*R], pCC[idx]);
	for(int i=0;i<R;i++) 
		pC[(idx*C)+i] = 0.0f;
    pCC[idx] = 0;
}

/*
 * Finalize kernel impl
 */
template <int R, int C>
__global__ void finalizeCentersBasic(float * pC, int * pCC) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= C) return;
    int nNumerator = pCC[idx];
    if(nNumerator == 0) {       
	    for(int i=0;i<R;i++) 
            pC[(idx*R)+i] = 0.0f;
    } else {
	    for(int i=0;i<R;i++) 
		    pC[(idx*R)+i] /= pCC[idx];
    }
}

template <int R, int C>
__global__ void finalizeCentersShmap(float * pC, int * pCC) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx >= R*C) return;
    int cidx = idx/R;
    // int nNumerator = pCC[idx];   TODO check
    int nNumerator = pCC[cidx];
    if(nNumerator == 0) {       
        pC[idx] = 0.0f;
    } else {
        pC[idx] /= pCC[cidx];
    }
}

template<int R, int C> 
__global__ void finalizeCentersColumnMajor(float * pC, int * pCC) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= C) return;
    
    int nNumerator = pCC[idx];
    if(nNumerator) {
	    for(int i=0;i<R;i++) {
		    pC[idx+(i*C)] /= pCC[idx];
        }
    } else {
	    for(int i=0;i<R;i++) {
		    pC[idx+(i*C)] = 0.0f;
        }
    }
}

template<int R, int C> 
__global__ void finalizeCentersColumnMajorShmap(float * pC, int * pCC) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int cidx = idx % C;
    if(cidx<C&&idx<R*C) {
        int nNumerator = pCC[cidx];
        if(nNumerator) {
            pC[idx] /= pCC[cidx];
        } else {
            pC[idx] = 0.0f;
        }
    }
}

/*
 * accum kernel impl
 */
template <int R, int C>
__global__ void accumulateCenters(float * pP, float * pC, int * pCC, int * pCI, int nP) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= nP) return;
	int clusterid = pCI[idx];
	for(int i=0;i<R;i++) 
		atomicAdd(&pC[(clusterid*R)+i], pP[(idx*R)+i]);
	atomicAdd(&pCC[clusterid], 1);
}

template<int R, int C> 
__global__ void accumulateCentersColumnMajor(float * pP, float * pC, int * pCC, int * pCI, int nP) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= nP) return;
	int clusterid = pCI[idx];
	for(int i=0;i<R;i++) {
		atomicAdd(&pC[clusterid+(i*C)], pP[idx+(i*nP)]);
	}
	atomicAdd(&pCC[clusterid], 1);
}

template<int R, int C> __global__
void accumulateSM_RCeqBlockSize(float * pP, float * pC, int * pCC, int * pCI, int nP) {
    __shared__ float accums[R*C];
    __shared__ int cnts[C];
    dassert(R*C*sizeof(float) <= ACCUM_SHMEMSIZE);
    dassert(C*sizeof(int) <= COUNTER_SHMEMSIZE);
    // dassert(R*C <= 1024);
    dassert(R*C <= MAXBLOCKTHREADS);
    dassert(threadIdx.x < R*C);
    if(threadIdx.x < R*C) accums[threadIdx.x] = 0.0f;
    if(threadIdx.x < C) cnts[threadIdx.x] = 0;
    __syncthreads();
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < nP) {
        int clusterid = pCI[idx];
        for(int i=0;i<R;i++) 
            atomicAdd(&accums[(clusterid*R)+i], pP[(idx*R)+i]);
        atomicAdd(&cnts[clusterid], 1);
    }
    __syncthreads();
    if(threadIdx.x < R*C) atomicAdd(&pC[threadIdx.x], accums[threadIdx.x]);
    if(threadIdx.x < C) atomicAdd(&pCC[threadIdx.x], cnts[threadIdx.x]);
}

template<int R, int C, int nLDElemsPerThread> __global__
void accumulateSM(float * pP, float * pC, int * pCC, int * pCI, int nP) {
    __shared__ float accums[R*C];
    __shared__ int cnts[C];
    dassert(R*C*sizeof(float) <= ACCUM_SHMEMSIZE);
    dassert(C*sizeof(int) <= COUNTER_SHMEMSIZE);
    if(threadIdx.x < C) cnts[threadIdx.x] = 0;
	for(int ridx=0; ridx<nLDElemsPerThread; ridx++) {
		int nCenterIdx = ridx*C;
		int nLDIdx = threadIdx.x + nCenterIdx;
        if(nLDIdx < R*C) accums[nLDIdx] = 0.0f;
    }
    __syncthreads();
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < nP) {
        int clusterid = pCI[idx];
        for(int i=0;i<R;i++) 
            atomicAdd(&accums[(clusterid*R)+i], pP[(idx*R)+i]);
        atomicAdd(&cnts[clusterid], 1);
    }
    __syncthreads();
    if(threadIdx.x < C) atomicAdd(&pCC[threadIdx.x], cnts[threadIdx.x]);
	for(int ridx=0; ridx<nLDElemsPerThread; ridx++) {
		int nCenterIdx = ridx*C;
		int nLDIdx = threadIdx.x + nCenterIdx;
        if(nLDIdx < R*C) 
			atomicAdd(&pC[nLDIdx], accums[nLDIdx]);
    }
}

template<int R, int C> 
__global__ void accumulateSMColumnMajor_RCeqBS(float * pP, float * pC, int * pCC, int * pCI, int nP) {
    __shared__ float accums[R*C];
    __shared__ int cnts[C];
    dassert(R*C*sizeof(float) <= ACCUM_SHMEMSIZE);
    dassert(C*sizeof(int) <= COUNTER_SHMEMSIZE);
    if(threadIdx.x < R*C) accums[threadIdx.x] = 0.0f;
    if(threadIdx.x < C) cnts[threadIdx.x] = 0;
    __syncthreads();
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < nP) {
        int clusterid = pCI[idx];
        for(int i=0;i<R;i++) 
            atomicAdd(&accums[clusterid+(C*i)], pP[idx+(nP*i)]);
        atomicAdd(&cnts[clusterid], 1);
    }
    __syncthreads();
    if(threadIdx.x < R*C) atomicAdd(&pC[threadIdx.x], accums[threadIdx.x]);
    if(threadIdx.x < C) atomicAdd(&pCC[threadIdx.x], cnts[threadIdx.x]);
}

template<int R, int C> 
__global__ void accumulateSMColumnMajor(float * pP, float * pC, int * pCC, int * pCI, int nP) {
    __shared__ float accums[R*C];
    __shared__ int cnts[C];
    dassert(R*C*sizeof(float) <= ACCUM_SHMEMSIZE);
    dassert(C*sizeof(int) <= COUNTER_SHMEMSIZE);
    if(threadIdx.x < C) {
        cnts[threadIdx.x] = 0;
        for(int i=0;i<R;i++) 
            accums[threadIdx.x*R+i] = 0.0f;
    }
    __syncthreads();
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < nP) {
        int clusterid = pCI[idx];
        for(int i=0;i<R;i++) 
            atomicAdd(&accums[clusterid+(C*i)], pP[idx+(nP*i)]);
        atomicAdd(&cnts[clusterid], 1);
    }
    __syncthreads();
    if(threadIdx.x < C) {
        atomicAdd(&pCC[threadIdx.x], cnts[threadIdx.x]);
        for(int i=0;i<R;i++) 
            atomicAdd(&pC[threadIdx.x*R+i], accums[threadIdx.x*R+i]);
    }
}

///////////////////////////////////////////////////////////////////////////////////
/// This the functions for mapping points to centers
///////////////////////////////////////////////////////////////////////////////////

template <int R, int C> 
__device__ float 
_vdistancef(float * a, float * b) {
    float accum = 0.0f;
    for(int i=0; i<R; i++) {
        float delta = a[i]-b[i];
        accum += delta*delta;
    }
    return sqrt(accum);
}

template<int R, int C>
__device__ float 
_vdistancefcm(
    int nAIndex,
    float * pAVectors,    
    int nAVectorCount,
    int nBIndex,
    float * pBVectors,
    int nBVectorCount
    ) 
{
    // assumes perfect packing 
    // (no trailing per-row pitch) 
    float accum = 0.0f;
    float * pAStart = &pAVectors[nAIndex];
    float * pBStart = &pBVectors[nBIndex];
    for(int i=0; i<R; i++) {
        float a = (*(pAStart + i*nAVectorCount));
        float b = (*(pBStart + i*nBVectorCount));
        float delta = a-b;
        accum += delta*delta;
    }
    return sqrt(accum);
}

template <int R, int C>
__device__ int 
nearestCenter(float * pP, float * pC) {
    float mindist = FLT_MAX;
    int minidx = 0;
	int clistidx = 0;
    for(int i=0; i<C;i++) {
		clistidx = i*R;
        float dist = _vdistancef<R,C>(pP, &pC[clistidx]);
        if(dist < mindist) {
            minidx = static_cast<int>(i);
            mindist = dist;
        }
    }
    return minidx;
}

template<int R, int C>
__device__ int 
nearestCenterColumnMajor(float * pP, float * pC, int nPointIndex, int nP){
    float mindist = FLT_MAX;
    int minidx = 0;
    for(int i=0; i<C;i++) {
        float dist = _vdistancefcm<R,C>(nPointIndex, pP, nP, i, pC, C);
        if(dist < mindist) {
            minidx = static_cast<int>(i);
            mindist = dist;
        }
    }
    return minidx;
}

template <int R, int C, bool bRO> 
__global__ void 
mapPointsToCenters(float * pP, float * pC, int * pCI, int nP) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= nP) return;
	pCI[idx] = nearestCenter<R,C>(&pP[idx*R], bRO ? d_cnst_centers : pC);
}



template<int R, int C, bool bRO> 
__global__ void 
mapPointsToCentersColumnMajor(float * pP, float * pC, int * pCI, int nP) {
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx >= nP) return;
	pCI[idx] = nearestCenterColumnMajor<R,C>(pP, bRO ? d_cnst_centers : pC, idx, nP);
}

/*
 * This the coop kernel impl
 */
template<int R, int C, int nLDElemsPerThread, bool bRO, bool ROWMAJ=true, bool sharedAccum=false, bool sharedFinalize=false,
        bool accumulateGeneral=false, bool accumulateSM=false>
__global__ void kmeansOnGPURaw(kmeans_params params) {
    /* Marshall off parameters */
    float * pP = params.pP;
    float * pC = params.pC;
    int * pCC = params.pCC;
    int * pCI = params.pCI;
    int nP = params.nP;

    int idx = blockIdx.x*blockDim.x+threadIdx.x;


    /* Map points to centers is the most time consuming part */
    /* mapPointsToCenters() and mapPointsToCentersColumnMajor() */
    grid_group grid = this_grid();
    {
        if (ROWMAJ) /* mapPointsToCenters()  */
        {
            if(idx < nP) {
                pCI[idx] = nearestCenter<R,C>(&pP[idx*R], bRO ? d_cnst_centers : pC);
            }
        } else {    /* mapPointsToCentersColumnMajor() */
            if(idx < nP) {
                pCI[idx] = nearestCenterColumnMajor<R,C>(pP, bRO ? d_cnst_centers : pC, idx, nP);
            }
        }
    }
    grid.sync();
    /* resetExplicit() and resetExplicitColumnMajor() */
    {
        if (ROWMAJ) {   /* resetExplicit() */
            if(idx < C) {
                for(int i=0;i<R;i++) 
                    pC[idx*R+i] = 0.0f;
                pCC[idx] = 0;
            }
        } else {    /* resetExplicitColumnMajor() */
            if(idx < C) {
                for(int i=0;i<R;i++) 
                    pC[(idx*C)+i] = 0.0f;
                pCC[idx] = 0;
            }
        }
    }
    grid.sync();
    {   /* accumulateCenters() and accumulateCentersColumnMajor() */
        if (sharedAccum == false) {
            if (ROWMAJ) {
                if(idx < nP) {
                int clusterid = pCI[idx];
                    for(int i=0;i<R;i++) 
                        atomicAdd(&pC[(clusterid*R)+i], pP[(idx*R)+i]);
                    atomicAdd(&pCC[clusterid], 1);
                }
            } else {    /* accumulateCentersColumnMajor() */
                if(idx < nP) {
                    int clusterid = pCI[idx];
                    for(int i=0;i<R;i++) {
                        atomicAdd(&pC[clusterid+(i*C)], pP[idx+(i*nP)]);
                    }
                    atomicAdd(&pCC[clusterid], 1);
                }
            }
        } else {
            if (accumulateGeneral == false) {
                if (ROWMAJ) {
                    /* accumulateSM_RCeqBlockSize() */
                    __shared__ float accums[R*C];
                    __shared__ int cnts[C];
                    dassert(R*C*sizeof(float) <= ACCUM_SHMEMSIZE);
                    dassert(C*sizeof(int) <= COUNTER_SHMEMSIZE);
                    // dassert(R*C <= 1024);
                    dassert(R*C <= MAXBLOCKTHREADS);
                    dassert(threadIdx.x < R*C);
                    if(threadIdx.x < R*C) accums[threadIdx.x] = 0.0f;
                    if(threadIdx.x < C) cnts[threadIdx.x] = 0;
                    __syncthreads();
                    if(idx < nP) {
                        int clusterid = pCI[idx];
                        for(int i=0;i<R;i++) 
                            atomicAdd(&accums[(clusterid*R)+i], pP[(idx*R)+i]);
                        atomicAdd(&cnts[clusterid], 1);
                    }
                    __syncthreads();
                    if(threadIdx.x < R*C) atomicAdd(&pC[threadIdx.x], accums[threadIdx.x]);
                    if(threadIdx.x < C) atomicAdd(&pCC[threadIdx.x], cnts[threadIdx.x]);
                } else {    /* accumulateSMColumnMajor_RCeqBS() */
                    __shared__ float accums[R*C];
                    __shared__ int cnts[C];
                    dassert(R*C*sizeof(float) <= ACCUM_SHMEMSIZE);
                    dassert(C*sizeof(int) <= COUNTER_SHMEMSIZE);
                    if(threadIdx.x < R*C) accums[threadIdx.x] = 0.0f;
                    if(threadIdx.x < C) cnts[threadIdx.x] = 0;
                    __syncthreads();
                    if(idx < nP) {
                        int clusterid = pCI[idx];
                        for(int i=0;i<R;i++) 
                            atomicAdd(&accums[clusterid+(C*i)], pP[idx+(nP*i)]);
                        atomicAdd(&cnts[clusterid], 1);
                    }
                    __syncthreads();
                    if(threadIdx.x < R*C) atomicAdd(&pC[threadIdx.x], accums[threadIdx.x]);
                    if(threadIdx.x < C) atomicAdd(&pCC[threadIdx.x], cnts[threadIdx.x]);
                }
            } else {
                if (ROWMAJ) {   /* accumulateSM() */
                    __shared__ float accums[R*C];
                    __shared__ int cnts[C];
                    dassert(R*C*sizeof(float) <= ACCUM_SHMEMSIZE);
                    dassert(C*sizeof(int) <= COUNTER_SHMEMSIZE);
                    if(threadIdx.x < C) cnts[threadIdx.x] = 0;
                    for(int ridx=0; ridx<nLDElemsPerThread; ridx++) {
                        int nCenterIdx = ridx*C;
                        int nLDIdx = threadIdx.x + nCenterIdx;
                        if(nLDIdx < R*C) accums[nLDIdx] = 0.0f;
                    }
                    __syncthreads();
                    if(idx < nP) {
                        int clusterid = pCI[idx];
                        for(int i=0;i<R;i++) 
                            atomicAdd(&accums[(clusterid*R)+i], pP[(idx*R)+i]);
                        atomicAdd(&cnts[clusterid], 1);
                    }
                    __syncthreads();
                    if(threadIdx.x < C) atomicAdd(&pCC[threadIdx.x], cnts[threadIdx.x]);
                    for(int ridx=0; ridx<nLDElemsPerThread; ridx++) {
                        int nCenterIdx = ridx*C;
                        int nLDIdx = threadIdx.x + nCenterIdx;
                        if(nLDIdx < R*C) 
                            atomicAdd(&pC[nLDIdx], accums[nLDIdx]);
                    }
                } else {    /* accumulateSMColumnMajor() */
                    __shared__ float accums[R*C];
                    __shared__ int cnts[C];
                    dassert(R*C*sizeof(float) <= ACCUM_SHMEMSIZE);
                    dassert(C*sizeof(int) <= COUNTER_SHMEMSIZE);
                    if(threadIdx.x < C) {
                        cnts[threadIdx.x] = 0;
                        for(int i=0;i<R;i++) 
                            accums[threadIdx.x*R+i] = 0.0f;
                    }
                    __syncthreads();
                    if(idx < nP) {
                        int clusterid = pCI[idx];
                        for(int i=0;i<R;i++) 
                            atomicAdd(&accums[clusterid+(C*i)], pP[idx+(nP*i)]);
                        atomicAdd(&cnts[clusterid], 1);
                    }
                    __syncthreads();
                    if(threadIdx.x < C) {
                        atomicAdd(&pCC[threadIdx.x], cnts[threadIdx.x]);
                        for(int i=0;i<R;i++) 
                            atomicAdd(&pC[threadIdx.x*R+i], accums[threadIdx.x*R+i]);
                    }
                }
            }
        }
    }
    grid.sync();
    /* finalizeCentersBasic() and finalizeCentersColumnMajor() */
    {   /* finalizeCentersBasic() */
        if (sharedFinalize == false ) {
            if (ROWMAJ) {
                if(idx >= C) return;
                int nNumerator = pCC[idx];
                if (nNumerator == 0) {       
                    for(int i=0;i<R;i++) 
                        pC[(idx*R)+i] = 0.0f;
                } else {
                    for(int i=0;i<R;i++) 
                        pC[(idx*R)+i] /= pCC[idx];
                }
            } else {    /* finalizeCentersColumnMajor() */
                if(idx >= C) return;
            
                int nNumerator = pCC[idx];
                if (nNumerator) {
                    for(int i=0;i<R;i++) {
                        pC[idx+(i*C)] /= pCC[idx];
                    }
                } else {
                    for(int i=0;i<R;i++) {
                        pC[idx+(i*C)] = 0.0f;
                    }
                }
            }
        } else {
            /* finalizeCentersShmap() */
            if (ROWMAJ) {
                if(idx >= R*C) return;
                int cidx = idx/R;
                // int nNumerator = pCC[idx];   TODO check
                int nNumerator = pCC[cidx];
                if(nNumerator == 0) {       
                    pC[idx] = 0.0f;
                } else {
                    pC[idx] /= pCC[cidx];
                }
            } else { /* finalizeCentersColumnMajorShmap() */
                int cidx = idx % C;
                if(cidx<C&&idx<R*C) {
                    int nNumerator = pCC[cidx];
                    if(nNumerator) {
                        pC[idx] /= pCC[cidx];
                    } else {
                        pC[idx] = 0.0f;
                    }
                }
            }
        }
    }
}