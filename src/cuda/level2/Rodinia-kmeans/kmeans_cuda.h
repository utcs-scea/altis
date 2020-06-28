////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\kmeans\kmeans_cuda.h
//
// summary:	Declares the kmeans cuda class
// 
// origin:  Rodinia(http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _KMEANS_CUDA_H_
#define _KMEANS_CUDA_H_

#include "ResultDatabase.h"

/// <summary>	delta -- had problems when return value was of float type. </summary>
int
kmeansCuda(float  **feature,				/* in: [npoints][nfeatures] */
           int      nfeatures,				/* number of attributes for each point */
           int      npoints,				/* number of data points */
           int      nclusters,				/* number of clusters */
           int     *membership,				/* which cluster the point belongs to */
		   float  **clusters,				/* coordinates of cluster centers */
		   int     *new_centers_len,		/* number of elements in each cluster */
           float  **new_centers,			/* sum of elements in each cluster */
           double &transferTime,
           double &kernelTime,
		   ResultDatabase &resultDB);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Allocate memory. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
///
/// <param name="npoints">  	The npoints. </param>
/// <param name="nfeatures">	The nfeatures. </param>
/// <param name="nclusters">	The nclusters. </param>
/// <param name="features"> 	[in,out] If non-null, the features. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void allocateMemory(int npoints, int nfeatures, int nclusters, float **features);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Deallocate memory. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

void deallocateMemory();

#endif
