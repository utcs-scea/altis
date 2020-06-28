////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\kmeans\kmeans_cuda.cu
//
// summary:	Kmeans cuda class
// 
// origin:  Rodinia(http://rodinia.cs.virginia.edu/doku.php)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>

#define THREADS_PER_DIM 16
#define BLOCKS_PER_DIM 16
#define THREADS_PER_BLOCK THREADS_PER_DIM*THREADS_PER_DIM

#include "cudacommon.h"
#include "ResultDatabase.h"
#include "OptionParser.h"
#include "kmeans_cuda_kernel.cu"

#include "kmeans.h"

//#define BLOCK_DELTA_REDUCE
//#define BLOCK_CENTER_REDUCE

#define CPU_DELTA_REDUCE
#define CPU_CENTER_REDUCE

int setup(ResultDatabase &resultDB, OptionParser &op);

// GLOBAL!!!!!
/* sqrt(256) -- see references for this choice */
unsigned int num_threads_perdim = THREADS_PER_DIM;					
/* temporary */
unsigned int num_blocks_perdim = BLOCKS_PER_DIM;
/* number of threads */
unsigned int num_threads = num_threads_perdim*num_threads_perdim;	
/* number of blocks */
unsigned int num_blocks = num_blocks_perdim*num_blocks_perdim;		

/* _d denotes it resides on the device */
/* newly assignment membership */
int    *membership_new;												
/* inverted data array */
float  *feature_d;													
/* original (not inverted) data array */
float  *feature_flipped_d;											
/* membership on the device */
int    *membership_d;												
/* sum of points in a cluster (per block) */
float  *block_new_centers;											
/* cluster centers on the device */
float  *clusters_d;													
/* per block calculation of cluster centers */
float  *block_clusters_d;											
/* per block calculation of deltas */
int    *block_deltas_d;												


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds a benchmark specifier options. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
///
/// <param name="op">	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("maxClusters", OPT_INT, "5", "maximum number of clusters allowed");
    op.addOption("minClusters", OPT_INT, "5", "minimum number of clusters allowed");
    op.addOption("threshold", OPT_FLOAT, "0.001", "threshold value");
    op.addOption("loops", OPT_INT, "1", "loop for each number of clusters");
    op.addOption("rmse", OPT_BOOL, "0", "calculate RMSE (default off)");
    op.addOption("outputCenters", OPT_BOOL, "0", "output cluster center coordinates (default off)");
    op.addOption("iterations", OPT_INT, "1000", "number of kmeans operation for each number of cluster in each loop");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    printf("Running KMeans\n");
    setup(resultDB, op);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	allocate device memory, calculate number of blocks and threads, and invert the data array. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
///
/// <param name="npoints">  	The npoints. </param>
/// <param name="nfeatures">	The nfeatures. </param>
/// <param name="nclusters">	The nclusters. </param>
/// <param name="features"> 	[in,out] If non-null, the features. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void allocateMemory(int npoints, int nfeatures, int nclusters, float **features)
{	
	num_blocks = npoints / num_threads;
	if (npoints % num_threads > 0)		/* defeat truncation */
		num_blocks++;

	num_blocks_perdim = sqrt((double) num_blocks);
	while (num_blocks_perdim * num_blocks_perdim < num_blocks)	// defeat truncation (should run once)
		num_blocks_perdim++;

	num_blocks = num_blocks_perdim*num_blocks_perdim;

	/* allocate memory for memory_new[] and initialize to -1 (host) */

    membership_new = (int*) malloc(npoints * sizeof(int));
    assert(membership_new);
	for(int i=0;i<npoints;i++) {
		membership_new[i] = -1;
	}

	/* allocate memory for block_new_centers[] (host) */
	block_new_centers = (float *) malloc(nclusters*nfeatures*sizeof(float));
    assert(block_new_centers);
    
	/* allocate memory for feature_flipped_d[][], feature_d[][] (device) */
	checkCudaErrors(cudaMalloc((void**) &feature_flipped_d, npoints*nfeatures*sizeof(float)));
	checkCudaErrors(cudaMemcpy(feature_flipped_d, features[0], npoints*nfeatures*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**) &feature_d, npoints*nfeatures*sizeof(float)));
		
	/* invert the data array (kernel execution) */	
	invert_mapping<<<num_blocks,num_threads>>>(feature_flipped_d,feature_d,npoints,nfeatures);
		
	/* allocate memory for membership_d[] and clusters_d[][] (device) */
	checkCudaErrors(cudaMalloc((void**) &membership_d, npoints*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**) &clusters_d, nclusters*nfeatures*sizeof(float)));
	
#ifdef BLOCK_DELTA_REDUCE
	// allocate array to hold the per block deltas on the gpu side
	checkCudaErrors(cudaMalloc((void**) &block_deltas_d, num_blocks_perdim * num_blocks_perdim * sizeof(int)));
	checkCudaErrors(cudaMemcpy(block_delta_d, &delta_h, sizeof(int), cudaMemcpyHostToDevice));
#endif

#ifdef BLOCK_CENTER_REDUCE
	// allocate memory and copy to card cluster  array in which to accumulate center points for the next iteration
    checkCudaErrors(cudaMalloc((void**) &block_clusters_d, 
            num_blocks_perdim * num_blocks_perdim * 
            nclusters * nfeatures * sizeof(float)));
    checkCudaErrors(cudaMemcpy(new_clusters_d, new_centers[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice));
#endif

}
/* -------------- allocateMemory() end ------------------- */


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	free host and device memory. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

void deallocateMemory()
{
	free(membership_new);
	free(block_new_centers);
    checkCudaErrors(cudaFree(membership_d));
	checkCudaErrors(cudaFree(clusters_d));
	checkCudaErrors(cudaFree(feature_d));
	//cudaFree(feature_flipped_d);
	//cudaFree(membership_d);

#ifdef BLOCK_CENTER_REDUCE
    checkCudaErrors(cudaFree(block_clusters_d));
#endif
#ifdef BLOCK_DELTA_REDUCE
    checkCudaErrors(cudaFree(block_deltas_d));
#endif
}
/* -------------- deallocateMemory() end ------------------- */


/* ------------------- kmeansCuda() ------------------------ */    

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Kmeans cuda. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
///
/// <param name="feature">		  	[in,out] If non-null, the feature. </param>
/// <param name="nfeatures">	  	The nfeatures. </param>
/// <param name="npoints">		  	The npoints. </param>
/// <param name="nclusters">	  	The nclusters. </param>
/// <param name="membership">	  	[in,out] If non-null, the membership. </param>
/// <param name="clusters">		  	[in,out] If non-null, the clusters. </param>
/// <param name="new_centers_len">	[in,out] If non-null, length of the new centers. </param>
/// <param name="new_centers">	  	[in,out] If non-null, the new centers. </param>
/// <param name="transferTime">   	[in,out] The transfer time. </param>
/// <param name="kernelTime">	  	[in,out] The kernel time. </param>
/// <param name="resultDB">		  	[in,out] The result database. </param>
///
/// <returns>	An int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

int	// delta -- had problems when return value was of float type
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
		   ResultDatabase &resultDB)
{
	int delta = 0;			/* if point has moved */
	int i,j;				/* counters */

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float elapsedTime;

    checkCudaErrors(cudaEventRecord(start, 0));
    /* copy membership (host to device) */
    checkCudaErrors(cudaMemcpy(membership_d, membership_new, npoints*sizeof(int), cudaMemcpyHostToDevice));
    /* copy clusters (host to device) */
    checkCudaErrors(cudaMemcpy(clusters_d, clusters[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    transferTime += elapsedTime * 1.e-3; // convert to seconds

    cudaError_t err;

	/* set up texture */
    cudaChannelFormatDesc chDesc0 = cudaCreateChannelDesc<float>();
    t_features.filterMode = cudaFilterModePoint;   
    t_features.normalized = false;
    t_features.channelDesc = chDesc0;

	err = cudaBindTexture(NULL, &t_features, feature_d, &chDesc0, npoints*nfeatures*sizeof(float));
    if (err != cudaSuccess) {
        printf("Error: Couldn't bind features array to texture, %d", err);
        exit(0);
    }

	cudaChannelFormatDesc chDesc1 = cudaCreateChannelDesc<float>();
    t_features_flipped.filterMode = cudaFilterModePoint;   
    t_features_flipped.normalized = false;
    t_features_flipped.channelDesc = chDesc1;

	err = cudaBindTexture(NULL, &t_features_flipped, feature_flipped_d, &chDesc1, npoints*nfeatures*sizeof(float));
    if (err != cudaSuccess) {
        printf("Error: Couldn't bind features_flipped array to texture, %d", err);
        exit(0);
    }
	// cudaChannelFormatDesc chDesc2 = cudaCreateChannelDesc<float>();
    // t_clusters.filterMode = cudaFilterModePoint;   
    // t_clusters.normalized = false;
    // t_clusters.channelDesc = chDesc2;

	// err = cudaBindTexture(NULL, &t_clusters, clusters_d, &chDesc2, nclusters*nfeatures*sizeof(float));
    // if(err != cudaSuccess) {
    //     printf("Error: Couldn't bind clusters array to texture, %d", err);
    //     exit(0);
    // }

    checkCudaErrors(cudaEventRecord(start, 0));
	/* copy clusters to constant memory */
	checkCudaErrors(cudaMemcpyToSymbol(c_clusters, clusters[0], 
                nclusters * nfeatures * sizeof(float),
                0, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    transferTime += elapsedTime * 1.e-3; // convert to seconds


    /* setup execution parameters.
	   changed to 2d (source code on NVIDIA CUDA Programming Guide) */
    dim3  grid( num_blocks_perdim, num_blocks_perdim );
    dim3  threads( num_threads_perdim*num_threads_perdim );
    
	/* execute the kernel */
    checkCudaErrors(cudaEventRecord(start, 0));
    kmeansPoint<<< grid, threads >>>( feature_d,
                                      nfeatures,
                                      npoints,
                                      nclusters,
                                      membership_d,
                                      clusters_d,
									  block_clusters_d,
									  block_deltas_d);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    kernelTime += elapsedTime * 1.e-3;

    checkCudaErrors(cudaEventRecord(start, 0));
	/* copy back membership (device to host) */
	checkCudaErrors(cudaMemcpy(membership_new, membership_d, npoints*sizeof(int), cudaMemcpyDeviceToHost));	
  
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    transferTime += elapsedTime * 1.e-3; // convert to seconds

#ifdef BLOCK_CENTER_REDUCE
    /*** Copy back arrays of per block sums ***/
    float * block_clusters_h = (float *) malloc(
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float));
        
    checkCudaErrors(cudaEventRecord(start, 0));

	checkCudaErrors(cudaMemcpy(block_clusters_h, block_clusters_d, 
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float), 
        cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    transferTime += elapsedTime * 1.e-3; // convert to seconds
#endif
#ifdef BLOCK_DELTA_REDUCE
    int * block_deltas_h = (int *) malloc(
        num_blocks_perdim * num_blocks_perdim * sizeof(int));
    assert(block_deltas_h);   
    checkCudaErrors(cudaEventRecord(start, 0));

	checkCudaErrors(cudaMemcpy(block_deltas_h, block_deltas_d, 
        num_blocks_perdim * num_blocks_perdim * sizeof(int), 
        cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    transferTime += elapsedTime * 1.e-3; // convert to seconds
#endif
    
	/* for each point, sum data points in each cluster
	   and see if membership has changed:
	     if so, increase delta and change old membership, and update new_centers;
	     otherwise, update new_centers */
	delta = 0;
	for (i = 0; i < npoints; i++)
	{		
		int cluster_id = membership_new[i];
		new_centers_len[cluster_id]++;
		if (membership_new[i] != membership[i])
		{
#ifdef CPU_DELTA_REDUCE
			delta++;
#endif
			membership[i] = membership_new[i];
		}
#ifdef CPU_CENTER_REDUCE
		for (j = 0; j < nfeatures; j++)
		{			
			new_centers[cluster_id][j] += feature[i][j];
		}
#endif
	}
	

#ifdef BLOCK_DELTA_REDUCE	
    /*** calculate global sums from per block sums for delta and the new centers ***/    
	
	//debug
    for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
        delta += block_deltas_h[i];
    }
        
#endif
#ifdef BLOCK_CENTER_REDUCE	
	
	for(int j = 0; j < nclusters;j++) {
		for(int k = 0; k < nfeatures;k++) {
			block_new_centers[j*nfeatures + k] = 0.f;
		}
	}

    for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
		for(int j = 0; j < nclusters;j++) {
			for(int k = 0; k < nfeatures;k++) {
				block_new_centers[j*nfeatures + k] += block_clusters_h[i * nclusters*nfeatures + j * nfeatures + k];
			}
		}
    }
	
#ifdef BLOCK_CENTER_REDUCE
	for(int j = 0; j < nclusters;j++) {
		for(int k = 0; k < nfeatures;k++)
			new_centers[j][k]= block_new_centers[j*nfeatures + k];		
	}
#endif

#endif

	return delta;
	
}
/* ------------------- kmeansCuda() end ------------------------ */    
