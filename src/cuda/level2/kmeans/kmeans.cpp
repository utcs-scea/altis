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
/**
 * **/
/**   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee
 * **/
/**				 University of Virginia
 * **/
/**
 * **/
/**   Description:	No longer supports fuzzy c-means clustering;
 * **/
/**					only regular k-means clustering.
 * **/
/**					No longer performs "validity" function to
 * analyze	**/
/**					compactness and separation crietria; instead
 * **/
/**					calculate root mean squared error.
 * **/
/**                                                                     **/
/*************************************************************************/
#define _CRT_SECURE_NO_DEPRECATE 1

#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "OptionParser.h"
#include "ResultDatabase.h"
#include "kmeans.h"

extern double wtime(void);

/*---< main() >-------------------------------------------------------------*/
int setup(ResultDatabase &resultDB, OptionParser &op) {
  const char *filename = op.getOptionString("inputFile").c_str();
  float *buf;
  char line[1024];
  bool isBinaryFile = op.getOptionBool("binaryInput");

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
  int nloops = op.getOptionInt("loops");

  bool isRMSE = op.getOptionBool("rmse");
  float rmse;

  bool isOutput = op.getOptionBool("outputCenters");

  /* ============== I/O begin ==============*/
  /* get nfeatures and npoints */
  // io_timing = omp_get_wtime();
  if (isBinaryFile) { // Binary file input
    int infile;
    if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
      fprintf(stderr, "Error: no such file (%s)\n", filename);
      exit(1);
    }
    read(infile, &npoints, sizeof(int));
    read(infile, &nfeatures, sizeof(int));

    /* allocate space for features[][] and read attributes of all objects */
    buf = (float *)malloc(npoints * nfeatures * sizeof(float));
    features = (float **)malloc(npoints * sizeof(float *));
    features[0] = (float *)malloc(npoints * nfeatures * sizeof(float));
    for (i = 1; i < npoints; i++)
      features[i] = features[i - 1] + nfeatures;

    read(infile, buf, npoints * nfeatures * sizeof(float));

    close(infile);
  } else {
    FILE *fp;
    string infile = op.getOptionString("inputFile");
    fp = fopen(infile.c_str(),"r");
    if(!fp)
    {
        printf("Error: Unable to read input file %s.\n", infile.c_str());
    }
    int n = fscanf(fp, "%d %d", &npoints, &nfeatures);
    
    printf("reading file\n");
    /*
    if ((infile = fopen(filename, "r")) == NULL) {
      fprintf(stderr, "Error: no such file (%s)\n", filename);
      exit(1);
    }
    while (fgets(line, 1024, infile) != NULL)
      if (strtok(line, " \t\n") != 0)
        npoints++;
    rewind(infile);
    while (fgets(line, 1024, infile) != NULL) {
      if (strtok(line, " \t\n") != 0) {
        ignore the id (first attribute): nfeatures = 1;
        while (strtok(NULL, " ,\t\n") != NULL)
          nfeatures++;
        break;
      }
    }*/

    /* allocate space for features[] and read attributes of all objects */
    buf = (float *)malloc(npoints * nfeatures * sizeof(float));
    features = (float **)malloc(npoints * sizeof(float *));
    features[0] = (float *)malloc(npoints * nfeatures * sizeof(float));
    for (i = 1; i < npoints; i++)
      features[i] = features[i - 1] + nfeatures;
    //rewind(fp);
    i = 0;
    while (fgets(line, 1024, fp) != NULL) {
      if (strtok(line, " \t\n") == NULL)
        continue;
      for (j = 0; j < nfeatures; j++) {
        buf[i] = atof(strtok(NULL, " ,\t\n"));
        i++;
      }
    }
    fclose(fp);
  }
  // io_timing = omp_get_wtime() - io_timing;

  printf("\nI/O completed\n");
  printf("\nNumber of objects: %d\n", npoints);
  printf("Number of features: %d\n", nfeatures);
  /* ============== I/O end ==============*/

  // error check for clusters
  if (npoints < min_nclusters) {
    printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n",
           min_nclusters, npoints);
    exit(0);
  }

  srand(7); /* seed for future random number generator */
  memcpy(
      features[0], buf,
      npoints * nfeatures *
          sizeof(
              float)); /* now features holds 2-dimensional array of features */
  free(buf);

  /* ======================= core of the clustering ===================*/

  // cluster_timing = omp_get_wtime();		/* Total clustering time */
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
                  nloops); /* number of iteration for each number of clusters */

  // cluster_timing = omp_get_wtime() - cluster_timing;

  /* =============== Command Line Output =============== */

  /* cluster center coordinates
     :displayed only for when k=1*/
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

  len = (float)((max_nclusters - min_nclusters + 1) * nloops);

  printf("Number of Iteration: %d\n", nloops);
  // printf("Time for I/O: %.5fsec\n", io_timing);
  // printf("Time for Entire Clustering: %.5fsec\n", cluster_timing);

  if (min_nclusters != max_nclusters) {
    if (nloops != 1) { // range of k, multiple iteration
      // printf("Average Clustering Time: %fsec\n",
      //		cluster_timing / len);
      printf("Best number of clusters is %d\n", best_nclusters);
    } else { // range of k, single iteration
      // printf("Average Clustering Time: %fsec\n",
      //		cluster_timing / len);
      printf("Best number of clusters is %d\n", best_nclusters);
    }
  } else {
    if (nloops != 1) { // single k, multiple iteration
      // printf("Average Clustering Time: %.5fsec\n",
      //		cluster_timing / nloops);
      if (isRMSE) // if calculated RMSE
        printf("Number of trials to approach the best RMSE of %.3f is %d\n",
               rmse, index + 1);
    } else {      // single k, single iteration
      if (isRMSE) // if calculated RMSE
        printf("Root Mean Squared Error: %.3f\n", rmse);
    }
  }

  /* free up memory */
  free(features[0]);
  free(features);
  return (0);
}
