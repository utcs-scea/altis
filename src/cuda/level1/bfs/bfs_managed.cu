void BFSGraphManagedMemory(ResultDatabase &resultDB, OptionParser &op, int no_of_nodes, int edge_list_size, int source, Node* &h_graph_nodes, int* &h_graph_edges) {
    bool verbose = op.getOptionBool("verbose");
	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;
	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

    // copy graph nodes to unified memory
    Node* graph_nodes;
    CUDA_SAFE_CALL(cudaMallocManaged(&graph_nodes, sizeof(Node)*no_of_nodes));
    memcpy(graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes);
    // copy graph edges to unified memory
    int* graph_edges;
    CUDA_SAFE_CALL(cudaMallocManaged(&graph_edges, sizeof(int)*edge_list_size));
    memcpy(graph_edges, h_graph_edges, sizeof(int)*edge_list_size);

	// allocate and initalize the memory
    bool* graph_mask, updating_graph_mask, graph_visited;
    CUDA_SAFE_CALL(cudaMallocManaged(&graph_mask, sizeof(bool)*no_of_nodes));
    CUDA_SAFE_CALL(cudaMallocManaged(&updating_graph_mask, sizeof(bool)*no_of_nodes));
    CUDA_SAFE_CALL(cudaMallocManaged(&graph_visited, sizeof(bool)*no_of_nodes));
    for( unsigned int i = 0; i < no_of_nodes; i++) 
    {
        graph_mask[i]=false;
        updating_graph_mask[i]=false;
        graph_visited[i]=false;
    }

	//set the source node as true in the mask
	graph_mask[source]=true;
	graph_visited[source]=true;

    // allocate and initialize memory for result
    int* cost;
    CUDA_SAFE_CALL(cudaMallocManaged(&cost, sizeof(int)*no_of_nodes);
	for(int i=0;i<no_of_nodes;i++) {
		cost[i]=-1;
    }
	cost[source]=0;

	// bool if execution is over
    bool* over;
    CUDA_SAFE_CALL(cudaMallocManaged(&over, sizeof(bool));

    // events for timing
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);
    float elapsedTime;

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

    double kernelTime = 0;
	int k=0;
	bool stop;
	//Call the Kernel untill all the elements of Frontier are not false
	do
	{
		//if no thread changes this value then the loop stops
		stop=false;
        stop = &over;

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

		k++;
	}
	while(stop);

    printf("Kernel Time: %f\n", kernelTime);
    printf("Kernel Executed %d times\n",k);

	cudaFree(graph_nodes);
	cudaFree(graph_edges);
	cudaFree(graph_mask);
	cudaFree(updating_graph_mask);
	cudaFree(graph_visited);
	cudaFree(cost);
    cudaFree(over);
}

