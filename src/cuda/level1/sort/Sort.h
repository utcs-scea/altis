////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level1\sort\Sort.h
//
// summary:	Declares the sort class
// 
// origin: SHOC Benchmark (https://github.com/vetter/shoc)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef SORT_H_
#define SORT_H_

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Defines an alias representing an unsigned integer. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

typedef unsigned int uint;

/// <summary>	Size of the sort block. </summary>
static const int SORT_BLOCK_SIZE = 128;
/// <summary>	Size of the scan block. </summary>
static const int SCAN_BLOCK_SIZE = 256;
/// <summary>	The sort bits. </summary>
static const int SORT_BITS = 32;


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Radix sort step. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="nbits">			The nbits. </param>
/// <param name="startbit">			The startbit. </param>
/// <param name="keys">				[in,out] If non-null, the keys. </param>
/// <param name="values">			[in,out] If non-null, the values. </param>
/// <param name="tempKeys">			[in,out] If non-null, the temporary keys. </param>
/// <param name="tempValues">   	[in,out] If non-null, the temporary values. </param>
/// <param name="counters">			[in,out] If non-null, the counters. </param>
/// <param name="countersSum">  	[in,out] If non-null, the counters sum. </param>
/// <param name="blockOffsets"> 	[in,out] If non-null, the block offsets. </param>
/// <param name="scanBlockSums">	[in,out] If non-null, the scan block sums. </param>
/// <param name="numElements">  	Number of elements. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
radixSortStep(uint nbits, uint startbit, uint4* keys, uint4* values,
        uint4* tempKeys, uint4* tempValues, uint* counters,
        uint* countersSum, uint* blockOffsets, uint** scanBlockSums,
        uint numElements);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Scans an array recursive. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="outArray">   	[in,out] If non-null, array of outs. </param>
/// <param name="inArray">	  	[in,out] If non-null, array of INS. </param>
/// <param name="numElements">	Number of elements. </param>
/// <param name="level">	  	The level. </param>
/// <param name="blockSums">  	[in,out] If non-null, the block sums. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
scanArrayRecursive(uint* outArray, uint* inArray, int numElements, int level,
        uint** blockSums);


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Verify sort correctness on cpu. </summary>
///
/// <remarks>	Ed, 5/19/2020. </remarks>
///
/// <param name="keys">   	[in,out] If non-null, the keys. </param>
/// <param name="vals">   	[in,out] If non-null, the vals. </param>
/// <param name="size">   	The size. </param>
/// <param name="verbose">	True to verbose. </param>
/// <param name="quiet">  	True to quiet. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

bool
verifySort(uint *keys, uint* vals, const size_t size, bool verbose, bool quiet);

#ifdef __DEVICE_EMULATION__

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines threads Synchronize. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define __SYNC __syncthreads();
#else

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Synchronize. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define __SYNC ;
#endif

#endif // SORT_H_
