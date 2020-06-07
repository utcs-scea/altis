#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"
#include <stdio.h>
#include <iostream>

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("verify", OPT_BOOL, "0", "verify the results computed on host");
    op.addOption("rank", OPT_INT, "24", "An integer-valued rank");
    op.addOption("centers", OPT_INT, "64", "An integer-valued centers argument");
    op.addOption("steps", OPT_INT, "100", "An integer-valued number of steps");
    op.addOption("type", OPT_STRING, "raw", "A valid version of kmeans");
}

void RunBenchmark(ResultDatabase &DB, OptionParser &op) {

    int PassFailFlag = 1;
}

