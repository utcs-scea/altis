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
/**   File:         example.c                                           **/
/**   Description:  Takes as input a file:                              **/
/**                 ascii  file: containing 1 data point per line       **/
/**                 binary file: first int is the number of objects     **/
/**                              2nd int is the no. of features of each **/
/**                              object                                 **/
/**                 This example performs a fuzzy c-means clustering    **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department Northwestern University                   **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**                                                                     **/
/**   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee                   **/
/**				 University of Virginia                                 **/
/**                                                                     **/
/**   Description:	No longer supports fuzzy c-means clustering;        **/
/**					only regular k-means clustering.                    **/
/**					No longer performs "validity" function to analyze	**/
/**					compactness and separation crietria; instead        **/
/**					calculate root mean squared error.                  **/
/**                                                                     **/

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	**********************************************************************. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define _CRT_SECURE_NO_DEPRECATE 1

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines seed for random number generator. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define SEED 7

#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "kmeans.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the wtime. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
///
/// <returns>	A double. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

extern double wtime(void);

/*---< main() >-------------------------------------------------------------*/

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Setups. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
///
/// <returns>	An int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

int setup(ResultDatabase &resultDB, OptionParser &op) {
    srand(SEED); /* seed for future random number generator */

    bool verbose = op.getOptionBool("verbose");
    bool quiet = op.getOptionBool("quiet");
    int nloops = op.getOptionInt("loops");
    int kmeansIterations = op.getOptionInt("iterations");
    bool isRMSE = op.getOptionBool("rmse");
    bool isOutput = op.getOptionBool("outputCenters");

    float *buf;
    char line[8192];

    float threshold = op.getOptionFloat("threshold");
    int max_nclusters = op.getOptionInt("maxClusters");
    int min_nclusters = op.getOptionInt("minClusters");
    int best_nclusters = 0;
    int nfeatures = 0;
    int npoints = 0;
    float len;
    float **features;
    float **cluster_centres = NULL;
    int i, j, index;
    float rmse;

    /* ============== I/O begin ==============*/
    // open file if filename is given
    FILE *fp = NULL;
    string infile = op.getOptionString("inputFile");
    if (infile.size() > 0) {
        fp = fopen(infile.c_str(),"r");
        if(!fp)
        {
            printf("Error: Unable to read input file %s.\n", infile.c_str());
            exit(0);
        } else {
            if (!quiet) {
                printf("Reading input file...");
            }
            while (fgets(line, 8192, fp) != NULL)
                if (strtok(line, " \t\n") != 0) npoints++;
            
            rewind(fp);
            while (fgets(line, 8192, fp) != NULL) {
                if (strtok(line, " \t\n") != 0) {
                    /* ignore the id (first attribute): nfeatures = 1; */
                    while (strtok(NULL, " ,\t\n") != NULL)
                        nfeatures++;
                    break;
                }
            }
        }
    } else {
        if(!quiet) {
            printf("Generating graph with a preset problem size %d", (int)op.getOptionInt("size"));
        }
        int npointsPresets[4] = {1, 10, 200, 200};
        npoints = npointsPresets[op.getOptionInt("size") - 1] * 10000;
        int nfeaturesPresets[4] = {10, 20, 35, 50};
        nfeatures = nfeaturesPresets[op.getOptionInt("size") - 1];
    }


    // allocate space for features[] and read attributes of all objects
    buf = (float *)malloc(npoints * nfeatures * sizeof(float));
    assert(buf);
    features = (float **)malloc(npoints * sizeof(float *));
    assert(features);
    features[0] = (float *)malloc(npoints * nfeatures * sizeof(float));
    assert(features[0]);
    // starting index for each point
    for (i = 1; i < npoints; i++) {
        features[i] = features[i - 1] + nfeatures;
    }

    rewind(fp);
    i = 0;
    // read/generate features for each point
    while (fgets(line, 8192, fp) != NULL) {
        if (strtok(line, " \t\n") == NULL) {
            continue;
        }
        for (int j = 0; j < nfeatures; j++) {
            if (fp){
                buf[i] = atof(strtok(NULL, " ,\t\n"));
            } else {
                buf[i] = rand() % 256;
            }
            i++;
        }
    }

    // close file
    if (fp) {
        fclose(fp);
    }

    if (!quiet) {
        printf("\nI/O completed\n");
        printf("\nNumber of objects: %d\n", npoints);
        printf("Number of features: %d\n", nfeatures);
    }
    /* ============== I/O end ==============*/

    // error check for clusters
    if (npoints < min_nclusters) {
        printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n",
            min_nclusters, npoints);
        exit(0);
    }

    memcpy(features[0], buf,
        npoints * nfeatures * sizeof(float)); /* now features holds 2-dimensional array of features */
    free(buf);

    /* ======================= core of the clustering ===================*/

    cluster_centres = NULL;
    index = cluster(npoints,       /* number of data points */
                    nfeatures,     /* number of features for each point */
                    features,      /* array: [npoints][nfeatures] */
                    min_nclusters, /* range of min to max number of clusters */
                    max_nclusters, threshold, /* loop termination factor */
                    &best_nclusters,  /* return: number between min and max */
                    &cluster_centres, /* return: [best_nclusters][nfeatures] */
                    &rmse,            /* Root Mean Squared Error */
                    isRMSE,           /* calculate RMSE */
                    nloops,
                    kmeansIterations,
                    resultDB,
                    quiet); /* number of iteration for each number of clusters */

    /* =============== Command Line Output =============== */

    /* cluster center coordinates
        :displayed only for when k=1*/
    if(!quiet) {
        if ((min_nclusters == max_nclusters) && (isOutput == 1)) {
            printf("\n================= Centroid Coordinates =================\n");
            for (i = 0; i < max_nclusters; i++) {
                printf("%d:", i);
                for (j = 0; j < nfeatures; j++) {
                    printf(" %.2f", cluster_centres[i][j]);
                }
                printf("\n\n");
            }
        }

        if (min_nclusters != max_nclusters) {
            printf("Best number of clusters is %d\n", best_nclusters);
        }
        if (isRMSE) { // if calculated RMSE
            printf("Root Mean Squared Error: %.3f\n", rmse);
        }
    }

    /* free up memory */
    free(features[0]);
    free(features);
    return (0);
}
