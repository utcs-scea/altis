////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level1\sort\sort_kernel.h
//
// summary:	Declares the sort kernel class
// 
// origin: SHOC ((https://github.com/vetter/shoc)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef SORT_KERNEL_H_
/// <summary>	. </summary>
#define SORT_KERNEL_H_

#include <cuda.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines warp size. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define WARP_SIZE 32

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines sort block size. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define SORT_BLOCK_SIZE 128

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines scan block size. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define SCAN_BLOCK_SIZE 256

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Defines an alias representing an unsigned integer. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

typedef unsigned int uint;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Radix sort blocks. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="nbits">		The nbits. </param>
/// <param name="startbit"> 	The startbit. </param>
/// <param name="keysOut">  	[in,out] If non-null, the keys out. </param>
/// <param name="valuesOut">	[in,out] If non-null, the values out. </param>
/// <param name="keysIn">   	[in,out] If non-null, the keys in. </param>
/// <param name="valuesIn"> 	[in,out] If non-null, the values in. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void radixSortBlocks(uint nbits, uint startbit, uint4* keysOut,
        uint4* valuesOut, uint4* keysIn, uint4* valuesIn);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Searches for the first radix offsets. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="keys">		   	[in,out] If non-null, the keys. </param>
/// <param name="counters">	   	[in,out] If non-null, the counters. </param>
/// <param name="blockOffsets">	[in,out] If non-null, the block offsets. </param>
/// <param name="startbit">	   	The startbit. </param>
/// <param name="numElements"> 	Number of elements. </param>
/// <param name="totalBlocks"> 	The total blocks. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void findRadixOffsets(uint2* keys, uint* counters,
        uint* blockOffsets, uint startbit, uint numElements, uint totalBlocks);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reorder data on device. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="startbit">	   	The startbit. </param>
/// <param name="outKeys">	   	[in,out] If non-null, the out keys. </param>
/// <param name="outValues">   	[in,out] If non-null, the out values. </param>
/// <param name="keys">		   	[in,out] If non-null, the keys. </param>
/// <param name="values">	   	[in,out] If non-null, the values. </param>
/// <param name="blockOffsets">	[in,out] If non-null, the block offsets. </param>
/// <param name="offsets">	   	[in,out] If non-null, the offsets. </param>
/// <param name="sizes">	   	[in,out] If non-null, the sizes. </param>
/// <param name="totalBlocks"> 	The total blocks. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void reorderData(uint startbit, uint *outKeys, uint *outValues,
        uint2 *keys, uint2 *values, uint *blockOffsets, uint *offsets,
        uint *sizes, uint totalBlocks);

// Scan Kernels

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Vector add, 4 elem per thread. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="d_vector">  	[in,out] If non-null, the vector. </param>
/// <param name="d_uniforms">	The uniforms. </param>
/// <param name="n">		 	An int to process. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void vectorAddUniform4(uint *d_vector, const uint *d_uniforms,
                                  const int n);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Scans. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="g_odata">	  	[in,out] If non-null, the odata. </param>
/// <param name="g_idata">	  	[in,out] If non-null, the idata. </param>
/// <param name="g_blockSums">	[in,out] If non-null, the block sums. </param>
/// <param name="n">		  	An int to process. </param>
/// <param name="fullBlock">  	True to full block. </param>
/// <param name="storeSum">   	True to store sum. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void scan(uint *g_odata, uint *g_idata, uint *g_blockSums,
        const int n, const bool fullBlock, const bool storeSum);

#endif // SORT_KERNEL_H_
