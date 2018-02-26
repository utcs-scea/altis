#include "ex_particle_CUDA_naive_seq.h"
#include "ex_particle_CUDA_float_seq.h"
#include "OptionParser.h"
#include "ResultDatabase.h"

void addBenchmarkSpecOptions(OptionParser &op) {
  //op.addOption("boxes1d", OPT_INT, "0",
  //             "specify number of boxes in single dimension, total box number is that^3");
}

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    particlefilter_naive(resultDB, op);
    /*
	printf("thread block size of kernel = %d \n", NUMBER_THREADS);
	// get boxes1d arg value
    int boxes1d = op.getOptionInt("boxes1d");
    if(boxes1d == 0) {
        int probSizes[4] = {10, 40, 100, 200};
        boxes1d = probSizes[op.getOptionInt("size") - 1];
    }
	printf("Configuration used: boxes1d = %d\n", boxes1d);

    int passes = op.getOptionInt("passes");
    for(int i = 0; i < passes; i++) {
        printf("Pass %d: ", i);
        runTest(resultDB, boxes1d);
        printf("Done.\n");
    }
    */
}
