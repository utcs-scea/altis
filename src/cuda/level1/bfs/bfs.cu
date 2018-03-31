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

int64_t* check;

struct Node
{
	int64_t starting;
	int64_t no_of_edges;
};

void initGraph(OptionParser &op, int64_t &no_of_nodes, int64_t &edge_list_size, int64_t &source, Node* &h_graph_nodes, int64_t* &h_graph_edges);
double BFSGraph(ResultDatabase &resultDB, OptionParser &op, int64_t no_of_nodes, int64_t edge_list_size, int64_t source, Node* &h_graph_nodes, int64_t* &h_graph_edges);
#ifdef UNIFIED_MEMORY
double BFSGraphUnifiedMemory(ResultDatabase &resultDB, OptionParser &op, int64_t no_of_nodes, int64_t edge_list_size, int64_t source, Node* &h_graph_nodes, int64_t* &h_graph_edges);
#endif

////////////////////////////////////////////////////////////////////////////////
__global__ void Kernel( Node* g_graph_nodes, int64_t* g_graph_edges, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int64_t* g_cost, int64_t no_of_nodes) 
{
	int64_t tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
    if(tid < 0) {
        return;
    }
	if( tid<no_of_nodes && g_graph_mask[tid])
	{
		g_graph_mask[tid]=false;
		for(int64_t i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++)
			{
			int64_t id = g_graph_edges[i];
			if(!g_graph_visited[id])
				{
				g_cost[id]=g_cost[tid]+1;
				g_updating_graph_mask[id]=true;
				}
			}
	}
}

__global__ void Kernel2( bool* g_graph_mask, bool *g_updating_graph_mask, bool* g_graph_visited, bool *g_over, int64_t no_of_nodes)
{
	int64_t tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
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

    int64_t no_of_nodes = 0;
    int64_t edge_list_size = 0;
    int64_t source = 0;
	Node* h_graph_nodes;
	int64_t* h_graph_edges;
    initGraph(op, no_of_nodes, edge_list_size, source, h_graph_nodes, h_graph_edges);

    // atts string for result database
    char tmp[64];
    sprintf(tmp, "%lldV,%lldE", no_of_nodes, edge_list_size);
    string atts = string(tmp);

    int64_t passes = op.getOptionInt("passes");
    for(int64_t i = 0; i < passes; i++) {
        printf("Pass %d:\n", i);
        printf("Executing BFS...");
        double time = BFSGraph(resultDB, op, no_of_nodes, edge_list_size, source, h_graph_nodes, h_graph_edges);
        printf("Done.\n");
#ifdef UNIFIED_MEMORY
        printf("Executing BFS using unified memory...");
        double timeUM = BFSGraphUnifiedMemory(resultDB, op, no_of_nodes, edge_list_size, source, h_graph_nodes, h_graph_edges);
        printf("Done.\n");
        resultDB.AddResult("Regular_Mem/Managed_Mem_Time", atts, "N", time/timeUM);
#endif
    }

	free( h_graph_nodes);
	free( h_graph_edges);
}

////////////////////////////////////////////////////////////////////////////////
//Generate uniform distribution
////////////////////////////////////////////////////////////////////////////////
int64_t uniform_distribution(int64_t rangeLow, int64_t rangeHigh) {
    double myRand = rand()/(1.0 + RAND_MAX); 
    int64_t range = rangeHigh - rangeLow + 1;
    int64_t myRand_scaled = (myRand * range) + rangeLow;
    return myRand_scaled;
}

////////////////////////////////////////////////////////////////////////////////
//Initialize Graph
////////////////////////////////////////////////////////////////////////////////
void initGraph(OptionParser &op, int64_t &no_of_nodes, int64_t &edge_list_size, int64_t &source, Node* &h_graph_nodes, int64_t* &h_graph_edges) {
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
	    int64_t n = fscanf(fp,"%lld",&no_of_nodes);
        assert(n == 1);
    } else {
        int64_t problemSizes[4] = {64, 256, 512, 1024};
        no_of_nodes = problemSizes[op.getOptionInt("size") - 1] * 1024 * 1024;
    }

	// initalize the nodes & number of edges
	h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	int64_t start;
    int64_t edgeno;   
    for( int64_t i = 0; i < no_of_nodes; i++) 
    {
        if(fp) {
            int64_t n = fscanf(fp,"%lld %d",&start,&edgeno);
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
	    int64_t n = fscanf(fp,"%lld",&source);
        assert(n == 1);
    } else {
        source = uniform_distribution(0, no_of_nodes - 1);
    }
    source = 0;

    if(fp) {
        int64_t edges;
        int64_t n = fscanf(fp,"%lld",&edges);
        assert(n == 1);
        assert(edges == edge_list_size);
    }

    // initialize the edges
	int64_t id;
    int64_t cost;
	h_graph_edges = (int64_t*) malloc(sizeof(int64_t)*edge_list_size);
	for(int64_t i=0; i < edge_list_size ; i++)
	{
        if(fp) {
            int64_t n = fscanf(fp,"%lld %d",&id, &cost);
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
double BFSGraph(ResultDatabase &resultDB, OptionParser &op, int64_t no_of_nodes, int64_t edge_list_size, int64_t source, Node* &h_graph_nodes, int64_t* &h_graph_edges) 
{
    bool verbose = op.getOptionBool("verbose");

	int64_t num_of_blocks = 1;
	int64_t num_of_threads_per_block = no_of_nodes;
	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int64_t)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	// allocate host memory
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);

	// initalize the memory
    for( int64_t i = 0; i < no_of_nodes; i++) 
    {
        h_graph_mask[i]=false;
        h_updating_graph_mask[i]=false;
        h_graph_visited[i]=false;
    }

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	// allocate mem for the result on host side
	int64_t* h_cost = (int64_t*) malloc( sizeof(int64_t)*no_of_nodes);
	for(int64_t i=0;i<no_of_nodes;i++) {
		h_cost[i]=-1;
    }
	h_cost[source]=0;

	// node list
	Node* d_graph_nodes;
	// edge list
	int64_t* d_graph_edges;
	// mask
	bool* d_graph_mask;
	bool* d_updating_graph_mask;
	// visited nodes
	bool* d_graph_visited;
    // result
	int64_t* d_cost;
	// bool if execution is over
	bool *d_over;

	cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
	cudaMalloc( (void**) &d_graph_edges, sizeof(int64_t)*edge_list_size) ;
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
	cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(int64_t)*edge_list_size, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
	cudaMemcpy( d_cost, h_cost, sizeof(int64_t)*no_of_nodes, cudaMemcpyHostToDevice) ;
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
	int64_t k=0;
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
	    printf("Kernel Executed %d times\n",k);
    }

	// copy result from device to host
    cudaEventRecord(tstart, 0);
	cudaMemcpy( h_cost, d_cost, sizeof(int64_t)*no_of_nodes, cudaMemcpyDeviceToHost) ;
    cudaEventRecord(tstop, 0);
    cudaEventSynchronize(tstop);
    cudaEventElapsedTime(&elapsedTime, tstart, tstop);
    transferTime += elapsedTime * 1.e-3; // convert to seconds

	//Store the result into a file
    string resultsfile = op.getOptionString("resultsfile");
    if(resultsfile != "") {
        FILE *fpo = fopen(resultsfile.c_str(),"w");
        for(int64_t i=0;i<no_of_nodes;i++) {
            fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
        }
        fclose(fpo);
        if(verbose) {
            printf("Result stored in %s\n", resultsfile.c_str());
        }
    }

    // store the result into an array to be compared with unified memory result
    check = (int64_t*) malloc(sizeof(int64_t)*no_of_nodes);
    for(int64_t i = 0; i < no_of_nodes; i++) {
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
    resultDB.AddResult("BFS-Rate_PCIe_Nodes", atts, "Nodes/s", no_of_nodes/(kernelTime + transferTime));
    resultDB.AddResult("BFS-Rate_PCIe_Edges", atts, "Edges/s", edge_list_size/(kernelTime + transferTime));
    resultDB.AddResult("BFS-Rate_Parity", atts, "N", transferTime / kernelTime);
    return transferTime + kernelTime;
}

#ifdef UNIFIED_MEMORY
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA and Unified Memory
////////////////////////////////////////////////////////////////////////////////
double BFSGraphUnifiedMemory(ResultDatabase &resultDB, OptionParser &op, int64_t no_of_nodes, int64_t edge_list_size, int64_t source, Node* &h_graph_nodes, int64_t* &h_graph_edges) {
    bool verbose = op.getOptionBool("verbose");
	int64_t num_of_blocks = 1;
	int64_t num_of_threads_per_block = no_of_nodes;
	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int64_t)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

    // copy graph nodes to unified memory
    Node* graph_nodes;
    CUDA_SAFE_CALL(cudaMallocManaged(&graph_nodes, sizeof(Node)*no_of_nodes));
    memcpy(graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes);
    // copy graph edges to unified memory
    int64_t* graph_edges;
    CUDA_SAFE_CALL(cudaMallocManaged(&graph_edges, sizeof(int64_t)*edge_list_size));
    memcpy(graph_edges, h_graph_edges, sizeof(int64_t)*edge_list_size);

	// allocate and initalize the memory
    bool* graph_mask;
    bool* updating_graph_mask;
    bool* graph_visited;
    CUDA_SAFE_CALL(cudaMallocManaged(&graph_mask, sizeof(bool)*no_of_nodes));
    CUDA_SAFE_CALL(cudaMallocManaged(&updating_graph_mask, sizeof(bool)*no_of_nodes));
    CUDA_SAFE_CALL(cudaMallocManaged(&graph_visited, sizeof(bool)*no_of_nodes));
    for( int64_t i = 0; i < no_of_nodes; i++) 
    {
        graph_mask[i]=false;
        updating_graph_mask[i]=false;
        graph_visited[i]=false;
    }

	//set the source node as true in the mask
	graph_mask[source]=true;
	graph_visited[source]=true;

    // allocate and initialize memory for result
    int64_t* cost;
    CUDA_SAFE_CALL(cudaMallocManaged(&cost, sizeof(int64_t)*no_of_nodes));
	for(int64_t i=0;i<no_of_nodes;i++) {
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
	int64_t k=0;
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
        printf("Kernel Executed %d times\n",k);
    }

    for(int64_t i = 0; i < no_of_nodes; i++) {
        if(cost[i] != check[i]) {
            printf("Error: Results don't match at index %d\n", i);
            break;
        }
    }

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
