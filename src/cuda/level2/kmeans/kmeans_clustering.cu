/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/ 
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/

/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department, Northwestern University                  **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee					**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					No longer performs "validity" function to analyze	**/
/**					compactness and separation crietria; instead		**/
/**					calculate root mean squared error.					**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>

#include "kmeans.h"
#include "kmeans_cuda.h"

#define RANDOM_MAX 2147483647

extern double wtime(void);

/*----< kmeans_clustering() >---------------------------------------------*/
float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          float   threshold,
                          int    *membership,
                          ResultDatabase &resultDB,
                          bool quiet) /* out: [npoints] */
{    
    int      i, j, n = 0;				/* counters */
	int		 loop=0, temp;
    int     *new_centers_len;	/* [nclusters]: no. of points in each cluster */
    float    delta;				/* if the point moved */
    float  **clusters;			/* out: [nclusters][nfeatures] */
    float  **new_centers;		/* [nclusters][nfeatures] */

	int     *initial;			/* used to hold the index of points not yet selected
								   prevents the "birthday problem" of dual selection (?)
								   considered holding initial cluster indices, but changed due to
								   possible, though unlikely, infinite loops */
	int      initial_points;
	int		 c = 0;

	/* nclusters should never be > npoints
	   that would guarantee a cluster without points */
	if (nclusters > npoints) {
		nclusters = npoints;
    }

    /* allocate space for and initialize returning variable clusters[] */
#ifdef UNIFIED_MEMORY
    CUDA_SAFE_CALL(cudaMallocManaged(&clusters, nclusters * sizeof(float*)));
    CUDA_SAFE_CALL(cudaMallocManaged(&(clusters[0]), nclusters* nfeatures * sizeof(float)));
#else
    clusters    = (float**) malloc(nclusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
#endif
    for (i=1; i<nclusters; i++)
        clusters[i] = clusters[i-1] + nfeatures;

	/* initialize the random clusters */
#ifdef UNIFIED_MEMORY
    CUDA_SAFE_CALL(cudaMallocManaged(&initial, npoints * sizeof(int)));
#else
	initial = (int *) malloc (npoints * sizeof(int));
#endif
	for (i = 0; i < npoints; i++) {
		initial[i] = i;
	}
	initial_points = npoints;

    /* randomly pick cluster centers */
    for (i=0; i<nclusters && initial_points >= 0; i++) {
		//n = (int)rand() % initial_points;		
        for (j=0; j<nfeatures; j++)
            clusters[i][j] = feature[initial[n]][j];	// remapped
		/* swap the selected index to the end (not really necessary,
		   could just move the end up) */
		temp = initial[n];
		initial[n] = initial[initial_points-1];
		initial[initial_points-1] = temp;
		initial_points--;
		n++;
    }

	/* initialize the membership to -1 for all */
    for (i=0; i < npoints; i++)
	  membership[i] = -1;

    /* allocate space for and initialize new_centers_len and new_centers */
    new_centers_len = (int*) calloc(nclusters, sizeof(int));

#ifdef UNIFIED_MEMORY
    CUDA_SAFE_CALL(cudaMallocManaged(&new_centers, nclusters * sizeof(float*)));
    CUDA_SAFE_CALL(cudaMallocManaged(&(new_centers[0]), nclusters * nfeatures * sizeof(float)));
#else
    new_centers    = (float**) malloc(nclusters *            sizeof(float*));
    new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
#endif
    for (i=1; i<nclusters; i++)
        new_centers[i] = new_centers[i-1] + nfeatures;

	/* iterate until convergence */
    do {
        delta = 0.0;
        // CUDA
        double transferTime = 0.;
        double kernelTime = 0;
        delta = (float) kmeansCuda(
                feature,		/* in: [npoints][nfeatures] */
                nfeatures,		/* number of attributes for each point */
                npoints,		/* number of data points */
                nclusters,		/* number of clusters */
                membership,		/* which cluster the point belongs to */
                clusters,		/* out: [nclusters][nfeatures] */
                new_centers_len,/* out: number of points in each cluster */
                new_centers,	/* sum of points in each cluster */
                transferTime,
                kernelTime,
                resultDB);
            
        char tmp[32];
        sprintf(tmp, "%dpoints,%dfeatures", npoints, nfeatures);
        string atts = string(tmp);
        resultDB.AddResult("KMeans-TransferTime", atts, "sec/iteration", transferTime);
        resultDB.AddResult("KMeans-KernelTime", atts, "sec/iteration", kernelTime);
        resultDB.AddResult("KMeans-TotalTime", atts, "sec/iteration", transferTime+kernelTime);
        resultDB.AddResult("KMeans-Rate_Parity", atts, "N", transferTime/kernelTime);
        resultDB.AddOverall("Time", "sec", kernelTime+transferTime);

		/* replace old cluster centers with new_centers */
		/* CPU side of reduction */
		for (i=0; i<nclusters; i++) {
			for (j=0; j<nfeatures; j++) {
				if (new_centers_len[i] > 0)
					clusters[i][j] = new_centers[i][j] / new_centers_len[i];	/* take average i.e. sum/n */
				new_centers[i][j] = 0.0;	/* set back to 0 */
			}
			new_centers_len[i] = 0;			/* set back to 0 */
		}	 
		c++;
    } while ((delta > threshold) && (loop++ < 500));	/* makes sure loop terminates */
    if(!quiet) {
        printf("Iterated %d times\n", c);
    }

#ifdef UNIFIED_MEMORY
    CUDA_SAFE_CALL(cudaFree(new_centers[0]));
    CUDA_SAFE_CALL(cudaFree(new_centers));
#else
    free(new_centers[0]);
    free(new_centers);
#endif
    free(new_centers_len);

    return clusters;
}

