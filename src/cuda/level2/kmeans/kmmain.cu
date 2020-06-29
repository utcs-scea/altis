#include "OptionParser.h"
#include "ResultDatabase.h"
#include "cudacommon.h"



#include <stdio.h>
#include <iostream>

#include "kmeansraw.h"


typedef double (*LPFNKMEANS)(ResultDatabase &DB,
                             const int nSteps,
                             void * h_Points,
                             void * h_Centers,
	                         const int nPoints,
	                         const int nCenters,
	                         bool bVerify,
	                         bool bVerbose);

typedef void (*LPFNBNC)(ResultDatabase &DB,
                        char * szFile, 
                        LPFNKMEANS lpfn, 
                        int nSteps,
                        int nSeed,
                        bool bVerify,
                        bool bVerbose);

#include "testsuitedecl.h"

//declare_suite_hdrs(4);
declare_suite_hdrs(16)
declare_suite_hdrs(24);
declare_suite_hdrs(32)
declare_suite_hdrs(64);
declare_suite_hdrs(128)


std::map<std::string, std::map<int, std::map<int, LPFNKMEANS>>> g_lpfns;
std::map<std::string, std::map<int, std::map<int, LPFNBNC>>> g_bncfns;
bool g_blpfnInit = false;

decl_init_lpfn_table_begin(g_lpfns, g_bncfns, g_blpfnInit);     // set g_blpfnInit to true

    //create_suite_entries(g_lpfns, g_bncfns, 4);
    create_suite_entries(g_lpfns, g_bncfns, 16);
    create_suite_entries(g_lpfns, g_bncfns, 24);
    create_suite_entries(g_lpfns, g_bncfns, 32);
    create_suite_entries(g_lpfns, g_bncfns, 64);
    create_suite_entries(g_lpfns, g_bncfns, 128);

decl_init_lpfn_table_end(g_lpfns, g_bncfns, g_blpfnInit);
declare_lpfn_finder(g_lpfns, g_blpfnInit)
declare_bnc_finder(g_bncfns, g_blpfnInit)

int gVerbose = 0;

bool g_bVerbose = false;
bool g_bVerify = false;
#define DEFAULTSTEPS 1000
int g_nSteps = DEFAULTSTEPS;


LPFNKMEANS g_lpfnKMeans = NULL;
LPFNBNC g_lpfnBnc = NULL;
char *g_lpszDefaultInput = "/home/ed/Desktop/altis/src/cuda/level2/km/inputs/random-n1000000-d128-c128.txt";
char g_vInputFile[4096];
char g_vKMeansVersion[4096];
int g_nRank;
int g_nCenters;
int g_nSeed = 0;

LPFNKMEANS
choose_kmeans_impl(
    char * lpszImpl,
    int nRank,
    int nCenters
    )
{
    // "raw" 
    // "constmem"
    // "constmemset"
    // "constmemsetshr"
    // "constmemsetshrmap"
    // "cm"
    // "cmconstmem"
    // "cmconstmemshr"
    // "cmconstmemshrmap"
    std::string strName(lpszImpl);
    return find_lpfn(strName, nRank, nCenters);
}

LPFNBNC
choose_kmeans_bnc(
    char * lpszImpl,
    int nRank,
    int nCenters
    )
{
    // "raw" 
    // "constmem"
    // "constmemset"
    // "constmemsetshr"
    // "constmemsetshrmap"
    // "cm"
    // "cmconstmem"
    // "cmconstmemshr"
    // "cmconstmemshrmap"
    std::string strName(lpszImpl);
    return find_bncfn(strName, nRank, nCenters);
}

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("verify", OPT_BOOL, "0", "verify the results computed on host");
    op.addOption("rank", OPT_INT, "16", "An integer-valued rank");
    op.addOption("centers", OPT_INT, "64", "An integer-valued centers argument");
    op.addOption("steps", OPT_INT, "1000", "An integer-valued number of steps");
    op.addOption("type", OPT_STRING, "raw", "A valid version of kmeans");
    op.addOption("seed", OPT_INT, "0", "seed for rand gen");
}

void RunBenchmark(ResultDatabase &DB, OptionParser &op) {
    int dev = op.getOptionInt("device");
    CUDA_SAFE_CALL(cudaSetDevice(dev));
    g_nRank = op.getOptionInt("rank");
    g_nCenters = op.getOptionInt("centers");
    g_nSteps = op.getOptionInt("steps");
    g_nSeed = op.getOptionInt("seed");
    strcpy(g_vKMeansVersion, op.getOptionString("type").c_str());
    if (g_nSeed == 0) {
        struct timespec ts;
        if (clock_gettime(CLOCK_MONOTONIC,&ts) != 0) {
            //error
            printf("failed to get clock tick\n");
            exit(-1);
        }
        g_nSeed = static_cast<int>(ts.tv_sec);
    } else {
        g_nSeed = op.getOptionInt("seed");
    }

    std::string inputFile = op.getOptionString("inputFile");


    g_lpfnKMeans = choose_kmeans_impl(g_vKMeansVersion, g_nRank, g_nCenters);
    g_lpfnBnc = choose_kmeans_bnc(g_vKMeansVersion, g_nRank, g_nCenters);
    if (!g_lpfnKMeans || !g_lpfnBnc) {
        if (!g_lpfnKMeans) std::cout <<"first" << std::endl;
        if (!g_lpfnBnc) std::cout <<"second" << std::endl;
	    fprintf(stderr, 
                "failed to select valid implementation for %s(RANK=%d, CENTERS=%d)!\n",
        g_vKMeansVersion,
        g_nRank,
        g_nCenters);
        exit(-1);
    }

    std::string customInput = op.getOptionString("inputFile");
    if (customInput.size() <= 0) {
        strncpy(g_vInputFile, g_lpszDefaultInput, 4096);
    } else {
        strncpy(g_vInputFile, customInput.c_str(), 4096);
    }

    (*g_lpfnBnc)(DB,
                 g_vInputFile,
                 g_lpfnKMeans,
                 g_nSteps,
                 g_nSeed,
                 g_bVerify,
                 g_bVerbose);
    
    cudaDeviceReset();
}

