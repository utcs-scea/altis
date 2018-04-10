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

#define MIN_NODES 20
#define MAX_NODES ULONG_MAX
#define MIN_EDGES 2
#define MAX_INIT_EDGES 4 // Nodes will have, on average, 2*MAX_INIT_EDGES edges
#define MIN_WEIGHT 1
#define MAX_WEIGHT 10
#define SEED 7

#define MAX_THREADS_PER_BLOCK 512

using namespace std;

long long* check;

struct Node
{
	long long starting;
	long long no_of_edges;
};

void initGraph(OptionParser &op, long long &no_of_nodes, long long &edge_list_size, long long &source, Node* &h_graph_nodes, long long* &h_graph_edges);
float BFSGraph(ResultDatabase &resultDB, OptionParser &op, long long no_of_nodes, long long edge_list_size, long long source, Node* &h_graph_nodes, long long* &h_graph_edges);
#ifdef UNIFIED_MEMORY
float BFSGraphUnifiedMemory(ResultDatabase &resultDB, OptionParser &op, long long no_of_nodes, long long edge_list_size, long long source, Node* &h_graph_nodes, long long* &h_graph_edges);
#endif

////////////////////////////////////////////////////////////////////////////////
__global__ void Kernel( Node* g_graph_nodes, long long* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, long long* g_cost, long long no_of_nodes) 
{
	long long tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
    if(tid < 0) {
        return;
    }
	if( tid<no_of_nodes && g_graph_mask[tid])
	{
		g_graph_mask[tid]=false;
		for(long long i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++)
			{
			long long id = g_graph_edges[i];
			if(!g_graph_visited[id])
				{
				g_cost[id]=g_cost[tid]+1;
				g_updating_graph_mask[id]=true;
				}
			}
	}
}

__global__ void Kernel2( bool* g_graph_mask, bool *g_updating_graph_mask, bool* g_graph_visited, bool *g_over, long long no_of_nodes)
{
	long long tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
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

	srand(SEED);

    long long no_of_nodes = 0;
    long long edge_list_size = 0;
    long long source = 0;
	Node* h_graph_nodes;
	long long* h_graph_edges;
    initGraph(op, no_of_nodes, edge_list_size, source, h_graph_nodes, h_graph_edges);

    // atts string for result database
    char tmp[64];
    sprintf(tmp, "%lldV,%lldE", no_of_nodes, edge_list_size);
    string atts = string(tmp);

    long long passes = op.getOptionInt("passes");
    for(long long i = 0; i < passes; i++) {
        printf("Pass %lld:\n", i);
        float time = BFSGraph(resultDB, op, no_of_nodes, edge_list_size, source, h_graph_nodes, h_graph_edges);
        if(time == FLT_MAX) {
            printf("Executing BFS...Out of Memory.\n");
        } else {
            printf("Executing BFS...Done.\n");
        }
#ifdef UNIFIED_MEMORY
        float timeUM = BFSGraphUnifiedMemory(resultDB, op, no_of_nodes, edge_list_size, source, h_graph_nodes, h_graph_edges);
        if(timeUM == FLT_MAX) {
            printf("Executing BFS using unified memory...Out of Memory.\n");
        } else {
            printf("Executing BFS using unified memory...Done.\n");
        }
        if(time != FLT_MAX && timeUM != FLT_MAX) {
            resultDB.AddResult("BFS_Time/BFS_UM_Time", atts, "N", time/timeUM);
        }
#endif
    }

	free( h_graph_nodes);
	free( h_graph_edges);
}

////////////////////////////////////////////////////////////////////////////////
//Generate uniform distribution
////////////////////////////////////////////////////////////////////////////////
long long uniform_distribution(long long rangeLow, long long rangeHigh) {
    double myRand = rand()/(1.0 + RAND_MAX); 
    long long range = rangeHigh - rangeLow + 1;
    long long myRand_scaled = (myRand * range) + rangeLow;
    return myRand_scaled;
}

////////////////////////////////////////////////////////////////////////////////
//Initialize Graph
////////////////////////////////////////////////////////////////////////////////
void initGraph(OptionParser &op, long long &no_of_nodes, long long &edge_list_size, long long &source, Node* &h_graph_nodes, long long* &h_graph_edges) {
    // open input file for reading
    FILE *fp = NULL;
    string infile = op.getOptionString("inputFile");
    if(infile != "") {
        fp = fopen(infile.c_str(),"r");
        if(!fp)
        {
            printf("Error: Unable to read graph file %s.\n", infile.c_str());
        }
    }

    if(fp) {
        printf("Reading graph file\n");
    } else {
        printf("Generating graph with a preset problem size\n");
    }

    // initialize number of nodes
    if(fp) {
	    long long n = fscanf(fp,"%lld",&no_of_nodes);
        assert(n == 1);
    } else {
        long long problemSizes[4] = {10, 50, 100, 400};
        no_of_nodes = problemSizes[op.getOptionInt("size") - 1] * 1024 * 1024;
    }

	// initalize the nodes & number of edges
	h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	long long start;
    long long edgeno;   
    for( long long i = 0; i < no_of_nodes; i++) 
    {
        if(fp) {
            long long n = fscanf(fp,"%lld %lld",&start,&edgeno);
            assert(n == 2);
        } else {
            start = edge_list_size;
            edgeno = rand() % (MAX_INIT_EDGES - MIN_EDGES + 1) + MIN_EDGES;
        }
        h_graph_nodes[i].starting = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        edge_list_size += edgeno;
    }

	// initialize the source node
    if(fp) {
	    long long n = fscanf(fp,"%lld",&source);
        assert(n == 1);
    } else {
        source = uniform_distribution(0, no_of_nodes - 1);
    }
    source = 0;

    if(fp) {
        long long edges;
        long long n = fscanf(fp,"%lld",&edges);
        assert(n == 1);
        assert(edges == edge_list_size);
    }

    // initialize the edges
	long long id;
    long long cost;
	h_graph_edges = (long long*) malloc(sizeof(long long)*edge_list_size);
	for(long long i=0; i < edge_list_size ; i++)
	{
        if(fp) {
            long long n = fscanf(fp,"%lld %lld",&id, &cost);
            assert(n == 2);
        } else {
			id = uniform_distribution(0, no_of_nodes - 1);
			//cost = rand() % (MAX_WEIGHT - MIN_WEIGHT + 1) + MIN_WEIGHT;
        }
		h_graph_edges[i] = id;
	}

    if(fp) {
        fclose(fp);    
        printf("Done reading graph file\n");
    } else {
        printf("Done generating graph\n");
    }

    printf("Graph size: %lld nodes, %lld edges\n", no_of_nodes, edge_list_size);
}

////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
float BFSGraph(ResultDatabase &resultDB, OptionParser &op, long long no_of_nodes, long long edge_list_size, long long source, Node* &h_graph_nodes, long long* &h_graph_edges) 
{
    bool verbose = op.getOptionBool("verbose");

	long long num_of_blocks = 1;
	long long num_of_threads_per_block = no_of_nodes;
	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (long long)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	// allocate host memory
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);

	// initalize the memory
    for( long long i = 0; i < no_of_nodes; i++) 
    {
        h_graph_mask[i]=false;
        h_updating_graph_mask[i]=false;
        h_graph_visited[i]=false;
    }

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	// allocate mem for the result on host side
	long long* h_cost = (long long*) malloc( sizeof(long long)*no_of_nodes);
	for(long long i=0;i<no_of_nodes;i++) {
		h_cost[i]=-1;
    }
	h_cost[source]=0;

	// node list
	Node* d_graph_nodes;
	// edge list
	long long* d_graph_edges;
	// mask
	bool* d_graph_mask;
	bool* d_updating_graph_mask;
	// visited nodes
	bool* d_graph_visited;
    // result
	long long* d_cost;
	// bool if execution is over
	bool *d_over;

	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_graph_edges, sizeof(long long)*edge_list_size));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_updating_graph_mask, sizeof(bool)*no_of_nodes));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_cost, sizeof(long long)*no_of_nodes));
	CUDA_SAFE_CALL(cudaMalloc( (void**) &d_over, sizeof(bool)));
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
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
        cudaFree(d_over);
        return FLT_MAX;
    }

    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);
    float elapsedTime;
double transferTime = 0.;
    cudaEventRecord(tstart, 0);
	cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(long long)*edge_list_size, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_cost, h_cost, sizeof(long long)*no_of_nodes, cudaMemcpyHostToDevice) ;
    cudaEventRecord(tstop, 0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    transferTime += elapsedTime * 1.e-3; // convert to seconds
    if(verbose) {
        printf("Transfer Time: %f\n", transferTime);
	    printf("Copied Everything to GPU memory\n");
    }

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

    if(verbose) {
	    printf("Start traversing the tree\n");
    }

    double kernelTime = 0;
	long long k=0;
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
        CHECK_CUDA_ERROR();

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
        printf("Kernel Time: %f\n", kernelTime);
	    printf("Kernel Executed %lld times\n",k);
    }

	// copy result from device to host
    cudaEventRecord(tstart, 0);
	cudaMemcpy( h_cost, d_cost, sizeof(long long)*no_of_nodes, cudaMemcpyDeviceToHost) ;
    cudaEventRecord(tstop, 0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    transferTime += elapsedTime * 1.e-3; // convert to seconds

	//Store the result into a file
    string resultsfile = op.getOptionString("resultsfile");
    if(resultsfile != "") {
        FILE *fpo = fopen(resultsfile.c_str(),"w");
        for(long long i=0;i<no_of_nodes;i++) {
            fprintf(fpo,"%lld) cost:%lld\n",i,h_cost[i]);
        }
        fclose(fpo);
        if(verbose) {
            printf("Result stored in %s\n", resultsfile.c_str());
        }
    }

    // store the result into an array to be compared with unified memory result
    check = (long long*) malloc(sizeof(long long)*no_of_nodes);
    for(long long i = 0; i < no_of_nodes; i++) {
        check[i] = h_cost[i];
    }

	// cleanup memory
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
    cudaFree(d_over);

    char tmp[64];
    sprintf(tmp, "%lldV,%lldE", no_of_nodes, edge_list_size);
    string atts = string(tmp);
    resultDB.AddResult("BFS-TransferTime", atts, "sec", transferTime);
    resultDB.AddResult("BFS-KernelTime", atts, "sec", kernelTime);
    resultDB.AddResult("BFS-TotalTime", atts, "sec", transferTime + kernelTime);
    resultDB.AddResult("BFS-Rate_Nodes", atts, "Nodes/s", no_of_nodes/kernelTime);
    resultDB.AddResult("BFS-Rate_Edges", atts, "Edges/s", edge_list_size/kernelTime);
    resultDB.AddResult("BFS-Rate_Parity", atts, "N", transferTime / kernelTime);
    return transferTime + kernelTime;
}

#ifdef UNIFIED_MEMORY
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA and Unified Memory
////////////////////////////////////////////////////////////////////////////////
float BFSGraphUnifiedMemory(ResultDatabase &resultDB, OptionParser &op, long long no_of_nodes, long long edge_list_size, long long source, Node* &h_graph_nodes, long long* &h_graph_edges) {
    bool verbose = op.getOptionBool("verbose");
	long long num_of_blocks = 1;
	long long num_of_threads_per_block = no_of_nodes;
	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (long long)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

    // copy graph nodes to unified memory
    Node* graph_nodes;
    CUDA_SAFE_CALL(cudaMallocManaged(&graph_nodes, sizeof(Node)*no_of_nodes));
    memcpy(graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes);
    // copy graph edges to unified memory
    long long* graph_edges;
    CUDA_SAFE_CALL(cudaMallocManaged(&graph_edges, sizeof(long long)*edge_list_size));
    memcpy(graph_edges, h_graph_edges, sizeof(long long)*edge_list_size);

	// allocate and initalize the memory
    bool* graph_mask;
    bool* updating_graph_mask;
    bool* graph_visited;
    CUDA_SAFE_CALL(cudaMallocManaged(&graph_mask, sizeof(bool)*no_of_nodes));
    CUDA_SAFE_CALL(cudaMallocManaged(&updating_graph_mask, sizeof(bool)*no_of_nodes));
    CUDA_SAFE_CALL(cudaMallocManaged(&graph_visited, sizeof(bool)*no_of_nodes));
    cudaError_t err = cudaGetLastError();
    for( long long i = 0; i < no_of_nodes; i++) 
    {
        graph_mask[i]=false;
        updating_graph_mask[i]=false;
        graph_visited[i]=false;
    }

	//set the source node as true in the mask
	graph_mask[source]=true;
	graph_visited[source]=true;

    // allocate and initialize memory for result
    long long* cost;
    CUDA_SAFE_CALL(cudaMallocManaged(&cost, sizeof(long long)*no_of_nodes));
    if(err != cudaSuccess) {
        free(check);
        cudaFree(graph_nodes);
        cudaFree(graph_edges);
        cudaFree(graph_mask);
        cudaFree(updating_graph_mask);
        cudaFree(graph_visited);
        cudaFree(cost);
        return FLT_MAX;
    }

	for(long long i=0;i<no_of_nodes;i++) {
		cost[i]=-1;
    }
	cost[source]=0;

	// bool if execution is over
    bool* over;
    CUDA_SAFE_CALL(cudaMallocManaged(&over, sizeof(bool)));

    // events for timing
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);
    float elapsedTime;

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

    double kernelTime = 0;
	long long k=0;
    bool stop;
	//Call the Kernel until all the elements of Frontier are not false
	do
	{
        stop = false;
        *over = stop;

        cudaEventRecord(tstart, 0);
        Kernel<<< grid, threads, 0 >>>(graph_nodes, graph_edges, graph_mask, updating_graph_mask, graph_visited, cost, no_of_nodes);
        cudaEventRecord(tstop, 0);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&elapsedTime, tstart, tstop);
        kernelTime += elapsedTime * 1.e-3;
        CHECK_CUDA_ERROR();

        // check if kernel execution generated an error
        cudaEventRecord(tstart, 0);
        Kernel2<<< grid, threads, 0 >>>(graph_mask, updating_graph_mask, graph_visited, over, no_of_nodes);
        cudaEventRecord(tstop, 0);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&elapsedTime, tstart, tstop);
        kernelTime += elapsedTime * 1.e-3;
        CHECK_CUDA_ERROR()

        stop = *over;
		k++;
	}
	while(stop);

    if(verbose) {
        printf("Kernel Time: %f\n", kernelTime);
        printf("Kernel Executed %lld times\n",k);
    }

    /*
    for(long long i = 0; i < no_of_nodes; i++) {
        if(cost[i] != check[i]) {
            printf("Error: Results don't match at index %lld\n", i);
            break;
        }
    }*/

    // cleanup memory
    free(check);
	cudaFree(graph_nodes);
	cudaFree(graph_edges);
	cudaFree(graph_mask);
	cudaFree(updating_graph_mask);
	cudaFree(graph_visited);
	cudaFree(cost);
    cudaFree(over);


    char tmp[64];
    sprintf(tmp, "%lldV,%lldE", no_of_nodes, edge_list_size);
    string atts = string(tmp);
    resultDB.AddResult("BFS-UM-TotalTime", atts, "sec", kernelTime);
    resultDB.AddResult("BFS-UM-Rate_Nodes", atts, "Nodes/s", no_of_nodes/kernelTime);
    resultDB.AddResult("BFS-UM-Rate_Edges", atts, "Edges/s", edge_list_size/kernelTime);
    return kernelTime;
}
#endif
