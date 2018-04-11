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

struct dwt {
    char * srcFilename;
    char * outFilename;
    unsigned char *srcImg;
    int pixWidth;
    int pixHeight;
    int components;
    int dwtLvls;
};

int getImg(char * srcFilename, unsigned char *srcImg, int inputSize)
{
    printf("Loading input: %s\n", srcFilename);

    int i = open(srcFilename, O_RDONLY, 0644);
    if (i == -1) { 
        error(0,errno,"Error: cannot access %s", srcFilename);
        return -1;
    }
    int ret = read(i, srcImg, inputSize);
    printf("precteno %d, inputsize %d\n", ret, inputSize);
    close(i);

    return 0;
}

template <typename T>
void processDWT(struct dwt *d, int forward, int writeVisual)
{
    int componentSize = d->pixWidth*d->pixHeight*sizeof(T); T *c_r_out, *backup ;
    CUDA_SAFE_CALL(cudaMalloc((void**)&c_r_out, componentSize)); //< aligned component size
    CUDA_SAFE_CALL(cudaMemset(c_r_out, 0, componentSize));
    
    CUDA_SAFE_CALL(cudaMalloc((void**)&backup, componentSize)); //< aligned component size
    CUDA_SAFE_CALL(cudaMemset(backup, 0, componentSize));
	
    if (d->components == 3) {
        /* Alloc two more buffers for G and B */
        T *c_g_out, *c_b_out;
        CUDA_SAFE_CALL(cudaMalloc((void**)&c_g_out, componentSize)); //< aligned component size
        CUDA_SAFE_CALL(cudaMemset(c_g_out, 0, componentSize));
        
        CUDA_SAFE_CALL(cudaMalloc((void**)&c_b_out, componentSize)); //< aligned component size
        CUDA_SAFE_CALL(cudaMemset(c_b_out, 0, componentSize));
        
        /* Load components */
        T *c_r, *c_g, *c_b;
        CUDA_SAFE_CALL(cudaMalloc((void**)&c_r, componentSize)); //< R, aligned component size
        CUDA_SAFE_CALL(cudaMemset(c_r, 0, componentSize));

        CUDA_SAFE_CALL(cudaMalloc((void**)&c_g, componentSize)); //< G, aligned component size
        CUDA_SAFE_CALL(cudaMemset(c_g, 0, componentSize));

        CUDA_SAFE_CALL(cudaMalloc((void**)&c_b, componentSize)); //< B, aligned component size
        CUDA_SAFE_CALL(cudaMemset(c_b, 0, componentSize));

        rgbToComponents(c_r, c_g, c_b, d->srcImg, d->pixWidth, d->pixHeight);
		
        /* Compute DWT and always store into file */
        nStage2dDWT(c_r, c_r_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
        nStage2dDWT(c_g, c_g_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
        nStage2dDWT(c_b, c_b_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);
     
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

        cudaFree(c_r);
        cudaFree(c_g);
        cudaFree(c_b);
        cudaFree(c_g_out);
        cudaFree(c_b_out);

    } 
    else if (d->components == 1) {
        //Load component
        T *c_r;
        CUDA_SAFE_CALL(cudaMalloc((void**)&(c_r), componentSize)); //< R, aligned component size
        CUDA_SAFE_CALL(cudaMemset(c_r, 0, componentSize));

        bwToComponent(c_r, d->srcImg, d->pixWidth, d->pixHeight);

        // Compute DWT 
        nStage2dDWT(c_r, c_r_out, backup, d->pixWidth, d->pixHeight, d->dwtLvls, forward);

        // Store DWT to file 
        if (writeVisual) {
            writeNStage2DDWT(c_r_out, d->pixWidth, d->pixHeight, d->dwtLvls, d->outFilename, ".out");
        } else {
            writeLinear(c_r_out, d->pixWidth, d->pixHeight, d->outFilename, ".lin.out");
        }
        cudaFree(c_r);
    }

    cudaFree(c_r_out);
    cudaFree(backup);
}

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("pixWidth", OPT_INT, "1", "real pixel width");
    op.addOption("pixHeight", OPT_INT, "1", "real pixel height");
    op.addOption("compCount", OPT_INT, "3", "number of components (3 for RGB/YUV, 4 for RGBA");
    op.addOption("bitDepth", OPT_INT, "8", "bit depth of src img");
    op.addOption("levels", OPT_INT, "3", "number of DWT levels");
    op.addOption("reverse", OPT_BOOL, "0", "reverse transform (defaults to forward");
    op.addOption("53", OPT_BOOL, "0", "5/3 transform (defaults to 9/7)");
    op.addOption("writeVisual", OPT_BOOL, "0", "write output in visual (tiled) order instead of linear");
}

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op)
{
    int pixWidth    = op.getOptionInt("pixWidth"); //<real pixWidth
    int pixHeight   = op.getOptionInt("pixHeight"); //<real pixHeight
    int compCount   = op.getOptionInt("compCount"); //number of components; 3 for RGB or YUV, 4 for RGBA
    int bitDepth    = op.getOptionInt("bitDepth");; 
    int dwtLvls     = op.getOptionInt("levels"); //default numuber of DWT levels
    bool forward     = !op.getOptionInt("reverse"); //forward transform
    bool dwt97       = !op.getOptionInt("53"); //1=dwt9/7, 0=dwt5/3 transform
    bool writeVisual = op.getOptionInt("writeVisual"); //write output (subbands) in visual (tiled) order instead of linear
    string inputFile = op.getOptionString("inputFile");

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
    printf("Source file:\t\t%s\n", d->srcFilename);
    printf(" Dimensions:\t\t%dx%d\n", pixWidth, pixHeight);
    printf(" Components count:\t%d\n", compCount);
    printf(" Bit depth:\t\t%d\n", bitDepth);
    printf(" DWT levels:\t\t%d\n", dwtLvls);
    printf(" Forward transform:\t%d\n", forward);
    printf(" 9/7 transform:\t\t%d\n", dwt97);
    printf(" Write visual:\t\t%d\n", writeVisual);
    
    //data sizes
    int inputSize = pixWidth*pixHeight*compCount; //<amount of data (in bytes) to proccess

    //load img source image
    cudaMallocHost((void **)&d->srcImg, inputSize);
    cudaCheckError("Alloc host memory");
    if (getImg(d->srcFilename, d->srcImg, inputSize) == -1) 
        return;

    /* DWT */
    if (forward == 1) {
        if(dwt97 == 1 )
            processDWT<float>(d, forward, writeVisual);
        else // 5/3
            processDWT<int>(d, forward, writeVisual);
    }
    else { // reverse
        if(dwt97 == 1 )
            processDWT<float>(d, forward, writeVisual);
        else // 5/3
            processDWT<int>(d, forward, writeVisual);
    }

    //writeComponent(r_cuda, pixWidth, pixHeight, srcFilename, ".g");
    //writeComponent(g_wave_cuda, 512000, ".g");
    //writeComponent(g_cuda, componentSize, ".g");
    //writeComponent(b_wave_cuda, componentSize, ".b");
    cudaFreeHost(d->srcImg);
    cudaCheckError("Cuda free host");
}
