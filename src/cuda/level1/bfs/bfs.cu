/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cassert>
#include <math.h>
#include <cuda.h>

#include "cudacommon.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

#define MAX_THREADS_PER_BLOCK 512

using namespace std;

int no_of_nodes;
int edge_list_size;
FILE *fp;

struct Node
{
	int starting;
	int no_of_edges;
};

void BFSGraph(ResultDatabase &resultDB, OptionParser &op);

////////////////////////////////////////////////////////////////////////////////
__global__ void Kernel( Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes) 
{
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if( tid<no_of_nodes && g_graph_mask[tid])
	{
		g_graph_mask[tid]=false;
		for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++)
			{
			int id = g_graph_edges[i];
			if(!g_graph_visited[id])
				{
				g_cost[id]=g_cost[tid]+1;
				g_updating_graph_mask[id]=true;
				}
			}
	}
}

__global__ void Kernel2( bool* g_graph_mask, bool *g_updating_graph_mask, bool* g_graph_visited, bool *g_over, int no_of_nodes)
{
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if( tid<no_of_nodes && g_updating_graph_mask[tid])
	{

		g_graph_mask[tid]=true;
		g_graph_visited[tid]=true;
		*g_over=true;
		g_updating_graph_mask[tid]=false;
	}
}
////////////////////////////////////////////////////////////////////////////////

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("resultsfile", OPT_STRING, "", "file to write results to");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the radix sort benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing, results are stored in resultDB
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    int passes = op.getOptionInt("passes");
    for(int i = 0; i < passes; i++) {
        printf("Pass %d: ", i);
        no_of_nodes=0;
        edge_list_size=0;
        BFSGraph(resultDB, op);
        printf("Done.\n");
    }
}

////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(ResultDatabase &resultDB, OptionParser &op) 
{
    bool verbose = op.getOptionBool("verbose");
    string infile = op.getOptionString("inputFile");

	//Read in Graph from a file
	fp = fopen(infile.c_str(),"r");
	if(!fp)
	{
		printf("Error: Unable to read graph file\n");
		return;
	}

    if(verbose) {
	    printf("Reading graph file\n");
    }

	fscanf(fp,"%d",&no_of_nodes);

	int source = 0;
	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);

	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
	int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp) {
		fclose(fp);    
    }
    if(verbose) {
	    printf("Done reading graph file\n");
    }

	// allocate mem for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	for(int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;

	// node list
	Node* d_graph_nodes;
	// edge list
	int* d_graph_edges;
	// mask
	bool* d_graph_mask;
	bool* d_updating_graph_mask;
	// visited nodes
	bool* d_graph_visited;
    // result
	int* d_cost;
	// bool if execution is over
	bool *d_over;

	cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
	cudaMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size) ;
	cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMalloc( (void**) &d_updating_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes) ;
	cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes);
	cudaMalloc( (void**) &d_over, sizeof(bool));

    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);
    float elapsedTime;

    double transferTime = 0.;
    cudaEventRecord(tstart, 0);
	cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice) ;
    cudaEventRecord(tstop, 0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    transferTime += elapsedTime * 1.e-3; // convert to seconds

    if(verbose) {
	    printf("Copied everything to device memory\n");
    }

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

    if(verbose) {
	    printf("Start traversing the tree\n");
    }
    double kernelTime = 0;
	int k=0;
	bool stop;
	//Call the Kernel untill all the elements of Frontier are not false
	do
	{
		//if no thread changes this value then the loop stops
		stop=false;
		cudaMemcpy( d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) ;

        cudaEventRecord(tstart, 0);
        Kernel<<< grid, threads, 0 >>>( d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);
        cudaEventRecord(tstop, 0);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&elapsedTime, tstart, tstop);
        kernelTime += elapsedTime * 1.e-3;
        CHECK_CUDA_ERROR()

        // check if kernel execution generated an error
        cudaEventRecord(tstart, 0);
        Kernel2<<< grid, threads, 0 >>>( d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);
        cudaEventRecord(tstop, 0);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&elapsedTime, tstart, tstop);
        kernelTime += elapsedTime * 1.e-3;
        CHECK_CUDA_ERROR()

        cudaMemcpy( &stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost) ;

		k++;
	}
	while(stop);

    if(verbose) {
	    printf("Kernel Executed %d times\n",k);
    }

	// copy result from device to host
    cudaEventRecord(tstart, 0);
	cudaMemcpy( h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost) ;
    cudaEventRecord(tstop, 0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    transferTime += elapsedTime * 1.e-3; // convert to seconds

	//Store the result into a file
    string resultsfile = op.getOptionString("resultsfile");
    if(resultsfile != "") {
        FILE *fpo = fopen(resultsfile.c_str(),"w");
        for(int i=0;i<no_of_nodes;i++) {
            fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
        }
        fclose(fpo);
        if(verbose) {
            printf("Result stored in %s\n", resultsfile.c_str());
        }
    }

	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
	cudaFree(d_graph_nodes);
	cudaFree(d_graph_edges);
	cudaFree(d_graph_mask);
	cudaFree(d_updating_graph_mask);
	cudaFree(d_graph_visited);
	cudaFree(d_cost);

    char atts[64];
    sprintf(atts, "%dV,%dE", no_of_nodes, edge_list_size);
    resultDB.AddResult("BFS-TransferTime", atts, "sec", transferTime);
    resultDB.AddResult("BFS-KernelTime", atts, "sec", kernelTime);
    resultDB.AddResult("BFS-Rate_Nodes", atts, "Nodes/s", no_of_nodes/kernelTime);
    resultDB.AddResult("BFS-Rate_Edges", atts, "Edges/s", edge_list_size/kernelTime);
    resultDB.AddResult("BFS-Rate_PCIe_Nodes", atts, "Nodes/s", no_of_nodes/(kernelTime + transferTime));
    resultDB.AddResult("BFS-Rate_PCIe_Edges", atts, "Edges/s", edge_list_size/(kernelTime + transferTime));
    resultDB.AddResult("BFS-Rate_Parity", atts, "N", transferTime / kernelTime);
}
