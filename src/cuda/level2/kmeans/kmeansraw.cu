///-------------------------------------------------------------------------------------------------
// file:	kmeansraw.cu
//
// summary:	kmeans implementation over extents of floats (no underlying point/vector struct)
///-------------------------------------------------------------------------------------------------


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <cuda_runtime.h>

#include "ResultDatabase.h"

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

#include "kmeansraw.h"
#include "testsuitedecl.h"

// declare_testsuite(4, 16);
// declare_testsuite(4, 32);
// declare_testsuite(4, 64);
// declare_testsuite(4, 128);
// declare_testsuite(4, 256);
// declare_testsuite(4, 512);
// 
declare_testsuite(16, 16);
declare_testsuite(16, 32);
declare_testsuite(16, 64);
declare_testsuite(16, 128);
declare_testsuite(16, 256);
declare_testsuite_lg(16, 512);
 
declare_testsuite(24, 16);
declare_testsuite(24, 32);
declare_testsuite(24, 64);
declare_testsuite(24, 128);
declare_testsuite_lg(24, 256);
declare_testsuite_lg(24, 512);
// 
declare_testsuite(32, 16);
declare_testsuite(32, 32);
declare_testsuite(32, 64);
declare_testsuite(32, 128);
declare_testsuite_lg(32, 256);
declare_testsuite_lg(32, 512);
// 
// declare_testsuite(64, 16);
// declare_testsuite(64, 32);
// declare_testsuite(64, 64);
// declare_testsuite(64, 128);
// declare_testsuite_lg(64, 256);
// declare_testsuite_lg(64, 512);
// 

// declare_testsuite(128, 16);
// declare_testsuite(128, 32);
// declare_testsuite(128, 64);
// declare_testsuite_lg(128, 128);
// declare_testsuite_lg(128, 256);
// declare_testsuite_lg(128, 512);
