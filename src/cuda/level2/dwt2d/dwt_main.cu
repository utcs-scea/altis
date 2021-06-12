/* 
 * Copyright (c) 2009, Jiri Matela
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // file:	altis\src\cuda\level2\dwt2d\dwt_cuda\dwt_main.cu
 //
 // summary:	Sort class
 // 
 // origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
 ////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cuda_profiler_api.h>

#include <unistd.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <getopt.h>

#include "ResultDatabase.h"
#include "OptionParser.h"
#include "cudacommon.h"
#include "common.h"
#include "components.h"
#include "dwt.h"
#include "data/create.cpp"

struct dwt {
    char * srcFilename;
    char * outFilename;
    unsigned char *srcImg;
    int pixWidth;
    int pixHeight;
    int components;
    int dwtLvls;
};

int getImg(char * srcFilename, unsigned char *srcImg, int inputSize, bool quiet)
{
    int i = open(srcFilename, O_RDONLY, 0644);
    if (i == -1) { 
        error(0,errno,"Error: cannot access %s", srcFilename);
        return -1;
    }
    int ret = read(i, srcImg, inputSize);
    close(i);

    if(!quiet) {
        printf("precteno %d, inputsize %d\n", ret, inputSize);
    }

    return 0;
}

template <typename T>
void processDWT(struct dwt *d, int forward, int writeVisual, ResultDatabase &resultDB, OptionParser &op, bool lastPass)
{
    // times
    float transferTime = 0;
    float kernelTime = 0;
    bool verbose = op.getOptionBool("verbose");
    bool quiet = op.getOptionBool("quiet");
    bool uvm = op.getOptionBool("uvm");

    int componentSize = d->pixWidth*d->pixHeight*sizeof(T); T *c_r_out, *backup ;
    if (uvm) {
        checkCudaErrors(cudaMallocManaged((void**)&c_r_out, componentSize));
    } else {
        checkCudaErrors(cudaMalloc((void**)&c_r_out, componentSize));
    }
    checkCudaErrors(cudaMemset(c_r_out, 0, componentSize));
    

#ifdef HYPERQ
        cudaStream_t streams[3];
/// <summary>	. </summary>
        for (int s = 0; s < 3; s++) {
            checkCudaErrors(cudaStreamCreate(&streams[s]));
        }
#endif

    if (uvm) {
#ifdef HYPERQ

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Gets the backup 3. </summary>
        ///
        /// <value>	The backup 3. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        T *backup2, *backup3;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="backup,componentSize">	[in,out] [in,out] If non-null, size of the backup,
        /// 									component. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        checkCudaErrors(cudaMallocManaged((void**)&backup, componentSize));

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="backup2,componentSize">	[in,out] [in,out] If non-null, size of the backup 2,
        /// 										component. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        checkCudaErrors(cudaMallocManaged((void**)&backup2, componentSize));

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="backup3,componentSize">	[in,out] [in,out] If non-null, size of the backup 3,
        /// 										component. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        checkCudaErrors(cudaMallocManaged((void**)&backup3, componentSize));

        // prefetch to increase performance
        //checkCudaErrors(cudaMemPrefetchAsync(backup, componentSize, 0, streams[0]));
        //checkCudaErrors(cudaMemPrefetchAsync(backup2, componentSize, 0, streams[1]));
        //checkCudaErrors(cudaMemPrefetchAsync(backup3, componentSize, 0, streams[2]));
#else

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="backup,componentSize">	[in,out] [in,out] If non-null, size of the backup,
        /// 									component. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        checkCudaErrors(cudaMallocManaged((void**)&backup, componentSize));
#endif
    } else {
#ifdef HYPERQ

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Gets the backup 3. </summary>
        ///
        /// <value>	The backup 3. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        T *backup2, *backup3;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="backup,componentSize">	[in,out] [in,out] If non-null, size of the backup,
        /// 									component. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        checkCudaErrors(cudaMalloc((void**)&backup, componentSize));

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="backup2,componentSize">	[in,out] [in,out] If non-null, size of the backup 2,
        /// 										component. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        checkCudaErrors(cudaMalloc((void**)&backup2, componentSize));

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="backup3,componentSize">	[in,out] [in,out] If non-null, size of the backup 3,
        /// 										component. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        checkCudaErrors(cudaMalloc((void**)&backup3, componentSize));
#else

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>	Constructor. </summary>
        ///
        /// <remarks>	Ed, 5/20/2020. </remarks>
        ///
        /// <param name="backup,componentSize">	[in,out] [in,out] If non-null, size of the backup,
        /// 									component. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        checkCudaErrors(cudaMalloc((void**)&backup, componentSize));
#endif
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>	Constructor. </summary>
    ///
    /// <remarks>	Ed, 5/20/2020. </remarks>
    ///
    /// <param name="parameter1">	The first parameter. </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    checkCudaErrors(cudaMemset(backup, 0, componentSize));
	
    if (d->components == 3) {


        /* Alloc two more buffers for G and B */
        T *c_g_out, *c_b_out;
        if (uvm) {
            checkCudaErrors(cudaMallocManaged((void**)&c_g_out, componentSize));
        } else {
            checkCudaErrors(cudaMalloc((void**)&c_g_out, componentSize));
        }
        checkCudaErrors(cudaMemset(c_g_out, 0, componentSize));
        
        if (uvm) {
            checkCudaErrors(cudaMallocManaged((void**)&c_b_out, componentSize));
        } else {
            checkCudaErrors(cudaMalloc((void**)&c_b_out, componentSize));
        }

        checkCudaErrors(cudaMemset(c_b_out, 0, componentSize));
        
        /* Load components */
        T *c_r, *c_g, *c_b;
        // R, aligned component size
        if (uvm) {
            checkCudaErrors(cudaMallocManaged((void**)&c_r, componentSize));
        } else {
            checkCudaErrors(cudaMalloc((void**)&c_r, componentSize));
        }

        checkCudaErrors(cudaMemset(c_r, 0, componentSize));
        // G, aligned component size

        if (uvm) {
            checkCudaErrors(cudaMallocManaged((void**)&c_g, componentSize));
        } else {
            checkCudaErrors(cudaMalloc((void**)&c_g, componentSize));
        }
        checkCudaErrors(cudaMemset(c_g, 0, componentSize));
        // B, aligned component size

        if (uvm) {
            checkCudaErrors(cudaMallocManaged((void**)&c_b, componentSize));
        } else {
            checkCudaErrors(cudaMalloc((void**)&c_b, componentSize));
        }
        checkCudaErrors(cudaMemset(c_b, 0, componentSize));

        rgbToComponents(c_r, c_g, c_b, d->srcImg, d->pixWidth, d->pixHeight, transferTime, kernelTime, op);
        /* Compute DWT and always store into file */
float time;
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);

cudaProfilerStart();
#ifdef HYPERQ
    // So far no concurrent kernel running, possible resource utilization
        nStage2dDWT(c_r, c_r_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward, transferTime, kernelTime, verbose, quiet, streams[0]);
        nStage2dDWT(c_g, c_g_out, backup2, d->pixWidth, d->pixHeight, d->dwtLvls, forward, transferTime, kernelTime, verbose, quiet, streams[1]);
        nStage2dDWT(c_b, c_b_out, backup3, d->pixWidth, d->pixHeight, d->dwtLvls, forward, transferTime, kernelTime, verbose, quiet, streams[2]);

#else
        nStage2dDWT(c_r, c_r_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward, transferTime, kernelTime, verbose, quiet);
        nStage2dDWT(c_g, c_g_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward, transferTime, kernelTime, verbose, quiet);
        nStage2dDWT(c_b, c_b_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward, transferTime, kernelTime, verbose, quiet);
#endif
cudaProfilerStop();

cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time, start, stop);
printf("Time to generate:  %3.1f ms \n", time);
        // -------test----------
        // T *h_r_out=(T*)malloc(componentSize);
		// cudaMemcpy(h_r_out, c_g_out, componentSize, cudaMemcpyDeviceToHost);
        // int ii;
		// for(ii=0;ii<componentSize/sizeof(T);ii++) {
			// fprintf(stderr, "%d ", h_r_out[ii]);
			// if((ii+1) % (d->pixWidth) == 0) fprintf(stderr, "\n");
        // }
        // -------test----------
        
		
        /* Store DWT to file */
        if (writeVisual) {
            writeNStage2DDWT(c_r_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".r");
            writeNStage2DDWT(c_g_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".g");
            writeNStage2DDWT(c_b_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".b");
        } else {
            writeLinear(c_r_out, d->pixWidth, d->pixHeight, d->outFilename, ".r");
            writeLinear(c_g_out, d->pixWidth, d->pixHeight, d->outFilename, ".g");
            writeLinear(c_b_out, d->pixWidth, d->pixHeight, d->outFilename, ".b");
        }
        if(lastPass && !quiet) {
            printf("Writing to %s.r (%d x %d)\n", d->outFilename, d->pixWidth, d->pixHeight);
            printf("Writing to %s.g (%d x %d)\n", d->outFilename, d->pixWidth, d->pixHeight);
            printf("Writing to %s.b (%d x %d)\n", d->outFilename, d->pixWidth, d->pixHeight);
        }
#ifdef HYPERQ
        for (int s = 0; s < 3; s++) {
            checkCudaErrors(cudaStreamDestroy(streams[s]));
        }
#endif
 
            

        cudaFree(c_r);
        cudaFree(c_g);
        cudaFree(c_b);
        cudaFree(c_g_out);
        cudaFree(c_b_out);

    } 
    else if (d->components == 1) {
        //Load component
        T *c_r;
        // R, aligned component size
        checkCudaErrors(cudaMalloc((void**)&(c_r), componentSize)); 
        checkCudaErrors(cudaMemset(c_r, 0, componentSize));

        bwToComponent(c_r, d->srcImg, d->pixWidth, d->pixHeight, transferTime, kernelTime);

        // Compute DWT 
        nStage2dDWT(c_r, c_r_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward, transferTime, kernelTime, verbose, quiet);

        // Store DWT to file 
        if (writeVisual) {
            writeNStage2DDWT(c_r_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".out");
            if(lastPass && !quiet) {
                printf("Writing to %s.out (%d x %d)\n", d->outFilename, d->pixWidth, d->pixHeight);
            }
        } else {
            writeLinear(c_r_out, d->pixWidth, d->pixHeight, d->outFilename, ".lin.out");
            if(lastPass && !quiet) {
                printf("Writing to %s.lin.out (%d x %d)\n", d->outFilename, d->pixWidth, d->pixHeight);
            }
        }
        cudaFree(c_r);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>	Constructor. </summary>
    ///
    /// <remarks>	Ed, 5/20/2020. </remarks>
    ///
    /// <param name="parameter1">	The first parameter. </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    cudaFree(c_r_out);

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>	Constructor. </summary>
    ///
    /// <remarks>	Ed, 5/20/2020. </remarks>
    ///
    /// <param name="parameter1">	The first parameter. </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    cudaFree(backup);

    /// <summary>	The atts[ 16]. </summary>
    char atts[16];

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>	Constructor. </summary>
    ///
    /// <remarks>	Ed, 5/20/2020. </remarks>
    ///
    /// <param name="parameter1">	The first parameter. </param>
    /// <param name="d"">		 	[in,out] The d". </param>
    /// <param name="parameter3">	The third parameter. </param>
    /// <param name="parameter4">	The fourth parameter. </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    sprintf(atts, "%dx%d", d->pixWidth, d->pixHeight);
    /// <summary>	The result db. add result. </summary>
    resultDB.AddResult("dwt_kernel_time", atts, "sec", kernelTime);
    /// <summary>	The result db. add result. </summary>
    resultDB.AddResult("dwt_transfer_time", atts, "sec", transferTime);
    /// <summary>	The result db. add result. </summary>
    resultDB.AddResult("dwt_total_time", atts, "sec", kernelTime+transferTime);
    /// <summary>	The result db. add result. </summary>
    resultDB.AddResult("dwt_parity", atts, "N", transferTime/kernelTime);
    /// <summary>	The result db. add overall. </summary>
    resultDB.AddOverall("Time", "sec", kernelTime+transferTime);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds a benchmark specifier options. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="op">	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("uvm", OPT_BOOL, "0", "enable CUDA Unified Virtual Memory, only use demand paging");
    op.addOption("pixWidth", OPT_INT, "1", "real pixel width");
    op.addOption("pixHeight", OPT_INT, "1", "real pixel height");
    op.addOption("compCount", OPT_INT, "3", "number of components (3 for RGB/YUV, 4 for RGBA");
    op.addOption("bitDepth", OPT_INT, "8", "bit depth of src img");
    op.addOption("levels", OPT_INT, "3", "number of DWT levels");
    op.addOption("reverse", OPT_BOOL, "0", "reverse transform (defaults to forward");
    op.addOption("53", OPT_BOOL, "0", "5/3 transform (defaults to 9/7)");
    op.addOption("writeVisual", OPT_BOOL, "0", "write output in visual (tiled) order instead of linear");
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{
    printf("Running DWT2D\n");
    bool quiet      = op.getOptionBool("quiet");
    bool verbose    = op.getOptionBool("verbose");
    int pixWidth    = op.getOptionInt("pixWidth"); //<real pixWidth
    int pixHeight   = op.getOptionInt("pixHeight"); //<real pixHeight
    int compCount   = op.getOptionInt("compCount"); //number of components; 3 for RGB or YUV, 4 for RGBA
    int bitDepth    = op.getOptionInt("bitDepth");; 
    int dwtLvls     = op.getOptionInt("levels"); //default numuber of DWT levels
    bool forward     = !op.getOptionBool("reverse"); //forward transform
    bool dwt97       = !op.getOptionBool("53"); //1=dwt9/7, 0=dwt5/3 transform
    bool writeVisual = op.getOptionBool("writeVisual"); //write output (subbands) in visual (tiled) order instead of linear
    string inputFile = op.getOptionString("inputFile");
    bool uvm = op.getOptionBool("uvm");
    if (inputFile.empty()) {
        int probSizes[4] = {48, 192, 8192, 2<<13};
        int pix = probSizes[op.getOptionInt("size")-1];
        inputFile = datagen(pix);
        pixWidth = pix;
        pixHeight = pix;
    }

    if (pixWidth <= 0 || pixHeight <=0) {
        printf("Wrong or missing dimensions\n");
        return;
    }

    if (forward == 0) {
        writeVisual = 0; //do not write visual when RDWT
    }

    struct dwt *d;
    d = (struct dwt *)malloc(sizeof(struct dwt));
    d->srcImg = NULL;
    d->pixWidth = pixWidth;
    d->pixHeight = pixHeight;
    d->components = compCount;
    d->dwtLvls  = dwtLvls;

    // file names
    d->srcFilename = (char*)malloc(strlen(inputFile.c_str()));
    strcpy(d->srcFilename, inputFile.c_str());
    d->outFilename = (char *)malloc(strlen(d->srcFilename)+4);
    strcpy(d->outFilename, d->srcFilename);
    strcpy(d->outFilename+strlen(d->srcFilename), ".dwt");

    //Input review
    if(!quiet) {
        printf("Source file:\t\t%s\n", d->srcFilename);
        printf(" Dimensions:\t\t%dx%d\n", pixWidth, pixHeight);
        printf(" Components count:\t%d\n", compCount);
        printf(" Bit depth:\t\t%d\n", bitDepth);
        printf(" DWT levels:\t\t%d\n", dwtLvls);
        printf(" Forward transform:\t%d\n", forward);
        printf(" 9/7 transform:\t\t%d\n", dwt97);
        printf(" Write visual:\t\t%d\n", writeVisual);
    }
    
    //data sizes
    int inputSize = pixWidth*pixHeight*compCount; //<amount of data (in bytes) to proccess

    //load img source image
    if (uvm) {
        checkCudaErrors(cudaMallocManaged((void **)&d->srcImg, inputSize));
    } else {
        checkCudaErrors(cudaMallocHost((void **)&d->srcImg, inputSize));
    }
    if (getImg(d->srcFilename, d->srcImg, inputSize, quiet) == -1) 
        return;

    int passes = op.getOptionInt("passes");
    for (int i = 0; i < passes; i++) {
        bool lastPass = i+1 == passes;
        if(!quiet) {
            printf("Pass %d:\n", i);
        }
        /* DWT */
        if (forward == 1) {
            if(dwt97 == 1 ) {
                processDWT<float>(d, forward, writeVisual, resultDB, op, lastPass);
            } else { // 5/3
                processDWT<int>(d, forward, writeVisual, resultDB, op, lastPass);
            }
        }
        else { // reverse
            if(dwt97 == 1 ) {
                processDWT<float>(d, forward, writeVisual, resultDB, op, lastPass);
            } else { // 5/3
                processDWT<int>(d, forward, writeVisual, resultDB, op, lastPass);
            }
        }
        if(!quiet) {
            printf("Done.\n");
        }
    }

    //writeComponent(r_cuda, pixWidth, pixHeight, srcFilename, ".g");
    //writeComponent(g_wave_cuda, 512000, ".g");
    //writeComponent(g_cuda, componentSize, ".g");
    //writeComponent(b_wave_cuda, componentSize, ".b");
    if (uvm) {
        checkCudaErrors(cudaFree(d->srcImg));
    } else {
        checkCudaErrors(cudaFreeHost(d->srcImg));
    }
}
