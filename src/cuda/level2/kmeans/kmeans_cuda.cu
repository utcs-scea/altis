#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include <cuda.h>

#define THREADS_PER_DIM 16
#define BLOCKS_PER_DIM 16
#define THREADS_PER_BLOCK THREADS_PER_DIM*THREADS_PER_DIM

#include "cudacommon.h"
#include "ResultDatabase.h"
#include "OptionParser.h"
#include "kmeans_cuda_kernel.cu"

//#define BLOCK_DELTA_REDUCE
//#define BLOCK_CENTER_REDUCE

#define CPU_DELTA_REDUCE
#define CPU_CENTER_REDUCE

int setup(ResultDatabase &resultDB, OptionParser &op);

// GLOBAL!!!!!
<<<<<<< HEAD
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
=======
unsigned int num_threads_perdim = THREADS_PER_DIM;					/* sqrt(256) -- see references for this choice */
unsigned int num_blocks_perdim = BLOCKS_PER_DIM;					/* temporary */
unsigned int num_threads = num_threads_perdim*num_threads_perdim;	/* number of threads */
unsigned int num_blocks = num_blocks_perdim*num_blocks_perdim;		/* number of blocks */

/* _d denotes it resides on the device */
int    *membership_new;												/* newly assignment membership */
float  *feature_d;													/* inverted data array */
float  *feature_flipped_d;											/* original (not inverted) data array */
int    *membership_d;												/* membership on the device */
float  *block_new_centers;											/* sum of points in a cluster (per block) */
float  *clusters_d;													/* cluster centers on the device */
float  *block_clusters_d;											/* per block calculation of cluster centers */
int    *block_deltas_d;												/* per block calculation of deltas */
>>>>>>> ac67b0ac89e7e88a81b0ae127701263ca7241fd2

////////////////////////////////////////////////////////////////////////////////
void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("maxClusters", OPT_INT, "5", "maximum number of clusters allowed");
    op.addOption("minClusters", OPT_INT, "5", "minimum number of clusters allowed");
    op.addOption("threshold", OPT_FLOAT, "0.001", "threshold value");
    op.addOption("loops", OPT_INT, "1", "iteration for each number of clusters");
    op.addOption("binaryInput", OPT_BOOL, "", "input file is in binary format");
    op.addOption("rmse", OPT_BOOL, "", "calculate RMSE (default off)");
    op.addOption("outputCenters", OPT_BOOL, "", "output cluster center coordinates (default off)");
}

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    setup(resultDB, op);
}
////////////////////////////////////////////////////////////////////////////////

/* -------------- allocateMemory() ------------------- */
/* allocate device memory, calculate number of blocks and threads, and invert the data array */
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
	for(int i=0;i<npoints;i++) {
		membership_new[i] = -1;
	}

	/* allocate memory for block_new_centers[] (host) */
	block_new_centers = (float *) malloc(nclusters*nfeatures*sizeof(float));
	
	/* allocate memory for feature_flipped_d[][], feature_d[][] (device) */
	cudaMalloc((void**) &feature_flipped_d, npoints*nfeatures*sizeof(float));
	cudaMemcpy(feature_flipped_d, features[0], npoints*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**) &feature_d, npoints*nfeatures*sizeof(float));
		
	/* invert the data array (kernel execution) */	
	invert_mapping<<<num_blocks,num_threads>>>(feature_flipped_d,feature_d,npoints,nfeatures);
<<<<<<< HEAD
    CHECK_CUDA_ERROR();
=======
>>>>>>> ac67b0ac89e7e88a81b0ae127701263ca7241fd2
		
	/* allocate memory for membership_d[] and clusters_d[][] (device) */
	cudaMalloc((void**) &membership_d, npoints*sizeof(int));
	cudaMalloc((void**) &clusters_d, nclusters*nfeatures*sizeof(float));

	
#ifdef BLOCK_DELTA_REDUCE
	// allocate array to hold the per block deltas on the gpu side
	
	cudaMalloc((void**) &block_deltas_d, num_blocks_perdim * num_blocks_perdim * sizeof(int));
	//cudaMemcpy(block_delta_d, &delta_h, sizeof(int), cudaMemcpyHostToDevice);
#endif

#ifdef BLOCK_CENTER_REDUCE
	// allocate memory and copy to card cluster  array in which to accumulate center points for the next iteration
<<<<<<< HEAD
    cudaMalloc((void**) &block_clusters_d, 
            num_blocks_perdim * num_blocks_perdim * 
            nclusters * nfeatures * sizeof(float));
=======
	cudaMalloc((void**) &block_clusters_d, 
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float));
>>>>>>> ac67b0ac89e7e88a81b0ae127701263ca7241fd2
	//cudaMemcpy(new_clusters_d, new_centers[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
#endif

}
/* -------------- allocateMemory() end ------------------- */

/* -------------- deallocateMemory() ------------------- */
/* free host and device memory */
void deallocateMemory()
{
	free(membership_new);
	free(block_new_centers);
	cudaFree(feature_d);
	cudaFree(feature_flipped_d);
	cudaFree(membership_d);

	cudaFree(clusters_d);
#ifdef BLOCK_CENTER_REDUCE
    cudaFree(block_clusters_d);
#endif
#ifdef BLOCK_DELTA_REDUCE
    cudaFree(block_deltas_d);
#endif
}
/* -------------- deallocateMemory() end ------------------- */


/* ------------------- kmeansCuda() ------------------------ */    
int	// delta -- had problems when return value was of float type
kmeansCuda(float  **feature,				/* in: [npoints][nfeatures] */
           int      nfeatures,				/* number of attributes for each point */
           int      npoints,				/* number of data points */
           int      nclusters,				/* number of clusters */
           int     *membership,				/* which cluster the point belongs to */
		   float  **clusters,				/* coordinates of cluster centers */
		   int     *new_centers_len,		/* number of elements in each cluster */
<<<<<<< HEAD
           float  **new_centers,			/* sum of elements in each cluster */
           double &transferTime,
           double &kernelTime,
		   ResultDatabase &resultDB)
=======
           float  **new_centers				/* sum of elements in each cluster */
		   )
>>>>>>> ac67b0ac89e7e88a81b0ae127701263ca7241fd2
{
	int delta = 0;			/* if point has moved */
	int i,j;				/* counters */

<<<<<<< HEAD
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;

    cudaEventRecord(start, 0);
    /* copy membership (host to device) */
    cudaMemcpy(membership_d, membership_new, npoints*sizeof(int), cudaMemcpyHostToDevice);
    /* copy clusters (host to device) */
    cudaMemcpy(clusters_d, clusters[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    transferTime += elapsedTime * 1.e-3; // convert to seconds

    cudaError_t err;
=======

	cudaSetDevice(1);

	/* copy membership (host to device) */
	cudaMemcpy(membership_d, membership_new, npoints*sizeof(int), cudaMemcpyHostToDevice);

	/* copy clusters (host to device) */
	cudaMemcpy(clusters_d, clusters[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
>>>>>>> ac67b0ac89e7e88a81b0ae127701263ca7241fd2

	/* set up texture */
    cudaChannelFormatDesc chDesc0 = cudaCreateChannelDesc<float>();
    t_features.filterMode = cudaFilterModePoint;   
    t_features.normalized = false;
    t_features.channelDesc = chDesc0;

<<<<<<< HEAD
	err = cudaBindTexture(NULL, &t_features, feature_d, &chDesc0, npoints*nfeatures*sizeof(float));
    if(err != CUDA_SUCCESS) {
        printf("Error: Couldn't bind features array to texture, %d", err);
    }
=======
	cudaBindTexture(NULL, &t_features, feature_d, &chDesc0, npoints*nfeatures*sizeof(float));
>>>>>>> ac67b0ac89e7e88a81b0ae127701263ca7241fd2

	cudaChannelFormatDesc chDesc1 = cudaCreateChannelDesc<float>();
    t_features_flipped.filterMode = cudaFilterModePoint;   
    t_features_flipped.normalized = false;
    t_features_flipped.channelDesc = chDesc1;

<<<<<<< HEAD
	err = cudaBindTexture(NULL, &t_features_flipped, feature_flipped_d, &chDesc1, npoints*nfeatures*sizeof(float));
    if(err != CUDA_SUCCESS) {
        printf("Error: Couldn't bind features_flipped array to texture, %d", err);
    }
=======
	cudaBindTexture(NULL, &t_features_flipped, feature_flipped_d, &chDesc1, npoints*nfeatures*sizeof(float));
>>>>>>> ac67b0ac89e7e88a81b0ae127701263ca7241fd2

	cudaChannelFormatDesc chDesc2 = cudaCreateChannelDesc<float>();
    t_clusters.filterMode = cudaFilterModePoint;   
    t_clusters.normalized = false;
    t_clusters.channelDesc = chDesc2;

<<<<<<< HEAD
	err = cudaBindTexture(NULL, &t_clusters, clusters_d, &chDesc2, nclusters*nfeatures*sizeof(float));
    if(err != CUDA_SUCCESS) {
        printf("Error: Couldn't bind clusters array to texture, %d", err);
    }

  cudaEventRecord(start, 0);
	/* copy clusters to constant memory */
	cudaMemcpyToSymbol("c_clusters",clusters[0],nclusters*nfeatures*sizeof(float),0,cudaMemcpyHostToDevice);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3; // convert to seconds
=======
	cudaBindTexture(NULL, &t_clusters, clusters_d, &chDesc2, nclusters*nfeatures*sizeof(float));

	/* copy clusters to constant memory */
	cudaMemcpyToSymbol("c_clusters",clusters[0],nclusters*nfeatures*sizeof(float),0,cudaMemcpyHostToDevice);
>>>>>>> ac67b0ac89e7e88a81b0ae127701263ca7241fd2


    /* setup execution parameters.
	   changed to 2d (source code on NVIDIA CUDA Programming Guide) */
    dim3  grid( num_blocks_perdim, num_blocks_perdim );
    dim3  threads( num_threads_perdim*num_threads_perdim );
    
	/* execute the kernel */
<<<<<<< HEAD
    cudaEventRecord(start, 0);
=======
>>>>>>> ac67b0ac89e7e88a81b0ae127701263ca7241fd2
    kmeansPoint<<< grid, threads >>>( feature_d,
                                      nfeatures,
                                      npoints,
                                      nclusters,
                                      membership_d,
                                      clusters_d,
									  block_clusters_d,
									  block_deltas_d);

	cudaThreadSynchronize();
<<<<<<< HEAD
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    kernelTime += elapsedTime * 1.e-3;
    //CHECK_CUDA_ERROR();

  cudaEventRecord(start, 0);
	/* copy back membership (device to host) */
	cudaMemcpy(membership_new, membership_d, npoints*sizeof(int), cudaMemcpyDeviceToHost);	
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3; // convert to seconds
=======

	/* copy back membership (device to host) */
	cudaMemcpy(membership_new, membership_d, npoints*sizeof(int), cudaMemcpyDeviceToHost);	
>>>>>>> ac67b0ac89e7e88a81b0ae127701263ca7241fd2

#ifdef BLOCK_CENTER_REDUCE
    /*** Copy back arrays of per block sums ***/
    float * block_clusters_h = (float *) malloc(
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float));
        
<<<<<<< HEAD
  cudaEventRecord(start, 0);
=======
>>>>>>> ac67b0ac89e7e88a81b0ae127701263ca7241fd2
	cudaMemcpy(block_clusters_h, block_clusters_d, 
        num_blocks_perdim * num_blocks_perdim * 
        nclusters * nfeatures * sizeof(float), 
        cudaMemcpyDeviceToHost);
<<<<<<< HEAD
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3; // convert to seconds
=======
>>>>>>> ac67b0ac89e7e88a81b0ae127701263ca7241fd2
#endif
#ifdef BLOCK_DELTA_REDUCE
    int * block_deltas_h = (int *) malloc(
        num_blocks_perdim * num_blocks_perdim * sizeof(int));
        
<<<<<<< HEAD
  cudaEventRecord(start, 0);
	cudaMemcpy(block_deltas_h, block_deltas_d, 
        num_blocks_perdim * num_blocks_perdim * sizeof(int), 
        cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3; // convert to seconds
=======
	cudaMemcpy(block_deltas_h, block_deltas_d, 
        num_blocks_perdim * num_blocks_perdim * sizeof(int), 
        cudaMemcpyDeviceToHost);
>>>>>>> ac67b0ac89e7e88a81b0ae127701263ca7241fd2
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
	//printf("\t \t reducing %d block sums to global sum \n",num_blocks_perdim * num_blocks_perdim);
    for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
		//printf("block %d delta is %d \n",i,block_deltas_h[i]);
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
	

#ifdef CPU_CENTER_REDUCE
	//debug
	/*for(int j = 0; j < nclusters;j++) {
		for(int k = 0; k < nfeatures;k++) {
			if(new_centers[j][k] >	1.001 * block_new_centers[j*nfeatures + k] || new_centers[j][k] <	0.999 * block_new_centers[j*nfeatures + k]) {
				printf("\t \t for %d:%d, normal value is %e and gpu reduced value id %e \n",j,k,new_centers[j][k],block_new_centers[j*nfeatures + k]);
			}
		}
	}*/
#endif

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
