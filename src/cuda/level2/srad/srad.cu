// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "srad.h"

#include "cudacommon.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

// includes, project
#include <cuda.h>

// includes, kernels
#include "srad_kernel.cu"

float kernelTime = 0.0f;
float transferTime = 0.0f;
cudaEvent_t start, stop;
float elapsed;

void random_matrix(float *I, int rows, int cols);
void runTest( int argc, char** argv);
void srad(ResultDatabase &resultDB, int imageSize, int speckleSize, int iters);

void addBenchmarkSpecOptions(OptionParser &op) {
    op.addOption("imageSize", OPT_INT, "0", "image height and width");
    op.addOption("speckleSize", OPT_INT, "0", "speckle height and width");
    op.addOption("iterations", OPT_INT, "0", "iterations of algorithm");
}

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) 
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
  int imageSize = op.getOptionInt("imageSize");
  int speckleSize = op.getOptionInt("speckleSize");
  int iters = op.getOptionInt("iterations");
  if(imageSize == 0 || speckleSize == 0 || iters == 0) {
      int imageSizes[4] = {128, 512, 4096, 8192};
      int iterSizes[4] = {5, 50, 100, 200};
      imageSize = imageSizes[op.getOptionInt("size") - 1];
      speckleSize = imageSize / 2;
      iters = iterSizes[op.getOptionInt("size") - 1];
  }
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  printf("Image Size: %d by %d\n", imageSize, imageSize);
  printf("Speckle size: %d by %d\n", speckleSize, speckleSize);
  printf("Num Iterations: %d\n", iters);

  int passes = op.getOptionInt("passes");
  for(int i = 0; i < passes; i++) {
      kernelTime = 0.0f;
      transferTime = 0.0f;
      printf("Pass %d:\n", i);
      srad(resultDB, imageSize, speckleSize, iters);
      printf("Done.\n");
      char atts[1024];
      sprintf(atts, "img:%d,speckle:%d,iter:%d", imageSize, speckleSize, iters);
      resultDB.AddResult("srad_kernel_time", atts, "sec", kernelTime);
      resultDB.AddResult("srad_transfer_time", atts, "sec", transferTime);
      resultDB.AddResult("srad_parity", atts, "N", transferTime / kernelTime);
  }
}


void srad(ResultDatabase &resultDB, int imageSize, int speckleSize, int iters) 
{
    int rows, cols, size_I, size_R, niter = 10, iter;
    float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI,varROI ;

#ifdef CPU
	float Jc, G2, L, num, den, qsqr;
	int *iN,*iS,*jE,*jW, k;
	float *dN,*dS,*dW,*dE;
	float cN,cS,cW,cE,D;
#endif

#ifdef GPU
	
	float *J_cuda;
    float *C_cuda;
	float *E_C, *W_C, *N_C, *S_C;

#endif

	unsigned int r1, r2, c1, c2;
	float *c;
    
	
 
    rows = imageSize;  //number of rows in the domain
    cols = imageSize;  //number of cols in the domain
    if ((rows%16!=0) || (cols%16!=0)){
        fprintf(stderr, "rows and cols must be multiples of 16\n");
        exit(1);
    }
    r1   = 0;  //y1 position of the speckle
    r2   = speckleSize;  //y2 position of the speckle
    c1   = 0;  //x1 position of the speckle
    c2   = speckleSize;  //x2 position of the speckle
    lambda = 0.5; //Lambda value
    niter = iters; //number of iterations


	size_I = cols * rows;
    size_R = (r2-r1+1)*(c2-c1+1);   

	I = (float *)malloc( size_I * sizeof(float) );
    J = (float *)malloc( size_I * sizeof(float) );
	c  = (float *)malloc(sizeof(float)* size_I) ;


#ifdef CPU

    iN = (int *)malloc(sizeof(unsigned int*) * rows) ;
    iS = (int *)malloc(sizeof(unsigned int*) * rows) ;
    jW = (int *)malloc(sizeof(unsigned int*) * cols) ;
    jE = (int *)malloc(sizeof(unsigned int*) * cols) ;    


	dN = (float *)malloc(sizeof(float)* size_I) ;
    dS = (float *)malloc(sizeof(float)* size_I) ;
    dW = (float *)malloc(sizeof(float)* size_I) ;
    dE = (float *)malloc(sizeof(float)* size_I) ;    
    

    for (int i=0; i< rows; i++) {
        iN[i] = i-1;
        iS[i] = i+1;
    }    
    for (int j=0; j< cols; j++) {
        jW[j] = j-1;
        jE[j] = j+1;
    }
    iN[0]    = 0;
    iS[rows-1] = rows-1;
    jW[0]    = 0;
    jE[cols-1] = cols-1;

#endif

#ifdef GPU

	//Allocate device memory
    CUDA_SAFE_CALL(cudaMalloc((void**)& J_cuda, sizeof(float)* size_I));
    CUDA_SAFE_CALL(cudaMalloc((void**)& C_cuda, sizeof(float)* size_I));
	CUDA_SAFE_CALL(cudaMalloc((void**)& E_C, sizeof(float)* size_I));
	CUDA_SAFE_CALL(cudaMalloc((void**)& W_C, sizeof(float)* size_I));
	CUDA_SAFE_CALL(cudaMalloc((void**)& S_C, sizeof(float)* size_I));
	CUDA_SAFE_CALL(cudaMalloc((void**)& N_C, sizeof(float)* size_I));

	
#endif 

	//Generate a random matrix
	random_matrix(I, rows, cols);

    for (int k = 0;  k < size_I; k++ ) {
     	J[k] = (float)exp(I[k]) ;
    }
 for (iter=0; iter< niter; iter++){     
		sum=0; sum2=0;
        for (int i=r1; i<=r2; i++) {
            for (int j=c1; j<=c2; j++) {
                tmp   = J[i * cols + j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / size_R;
        varROI  = (sum2 / size_R) - meanROI*meanROI;
        q0sqr   = varROI / (meanROI*meanROI);

#ifdef CPU
        
		for (int i = 0 ; i < rows ; i++) {
            for (int j = 0; j < cols; j++) { 
		
				k = i * cols + j;
				Jc = J[k];
 
				// directional derivates
                dN[k] = J[iN[i] * cols + j] - Jc;
                dS[k] = J[iS[i] * cols + j] - Jc;
                dW[k] = J[i * cols + jW[j]] - Jc;
                dE[k] = J[i * cols + jE[j]] - Jc;
			
                G2 = (dN[k]*dN[k] + dS[k]*dS[k] 
                    + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

   		        L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

				num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
                den  = 1 + (.25*L);
                qsqr = num/(den*den);
 
                // diffusion coefficent (equ 33)
                den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
                c[k] = 1.0 / (1.0+den) ;
                
                // saturate diffusion coefficent
                if (c[k] < 0) {c[k] = 0;}
                else if (c[k] > 1) {c[k] = 1;}
		}
	}
         for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {        

                // current index
                k = i * cols + j;
                
                // diffusion coefficent
					cN = c[k];
					cS = c[iS[i] * cols + j];
					cW = c[k];
					cE = c[i * cols + jE[j]];

                // divergence (equ 58)
                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
                
                // image update (equ 61)
                J[k] = J[k] + 0.25*lambda*D;
            }
	}

#endif // CPU


#ifdef GPU

	//Currently the input size must be divided by 16 - the block size
	int block_x = cols/BLOCK_SIZE ;
    int block_y = rows/BLOCK_SIZE ;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(block_x , block_y);
    

	//Copy data from main memory to device memory
    cudaEventRecord(start, 0);
	CUDA_SAFE_CALL(cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    transferTime += elapsed * 1.e-3;

	//Run kernels
    cudaEventRecord(start, 0);
	srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr); 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();

    cudaEventRecord(start, 0);
	srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr); 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();

	//Copy data from device memory to main memory
    cudaEventRecord(start, 0);
    CUDA_SAFE_CALL(cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    transferTime += elapsed * 1.e-3;

#endif   
}

    cudaThreadSynchronize();

#ifdef OUTPUT
    //Printing output	
		printf("Printing Output:\n"); 
    for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
         printf("%.5f ", J[i * cols + j]); 
		}	
     printf("\n"); 
   }
#endif 


	free(I);
	free(J);
#ifdef CPU
	free(iN); free(iS); free(jW); free(jE);
    free(dN); free(dS); free(dW); free(dE);
#endif
#ifdef GPU
    CUDA_SAFE_CALL(cudaFree(C_cuda));
	CUDA_SAFE_CALL(cudaFree(J_cuda));
	CUDA_SAFE_CALL(cudaFree(E_C));
	CUDA_SAFE_CALL(cudaFree(W_C));
	CUDA_SAFE_CALL(cudaFree(N_C));
	CUDA_SAFE_CALL(cudaFree(S_C));
#endif 
	free(c);
  
}


void random_matrix(float *I, int rows, int cols){
    
	srand(7);
	
	for( int i = 0 ; i < rows ; i++){
		for ( int j = 0 ; j < cols ; j++){
		 I[i * cols + j] = rand()/(float)RAND_MAX ;
		}
	}

}
