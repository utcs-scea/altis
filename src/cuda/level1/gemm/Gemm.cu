////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	C:\Users\ed\source\repos\altis\src\cuda\level1\gemm\Gemm.cu
//
// summary:	Gemm class
// 
// origin: SHOC (https://github.com/vetter/shocp)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OptionParser.h"
#include "ResultDatabase.h"
// #include "Timer.h"
#include "Utility.h"
//#include "cublas.h"
#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cudacommon.h"
#include "cuda_fp16.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define SEED 7
/// <summary>	Length of the object field. </summary>
static const int FIELD_LENGTH = 128;

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the test operation. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="testName">	Name of the test. </param>
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op);

// origianlly don't need handle in v1 cublas

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gemm operation wrapper. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="transa">	The transa. </param>
/// <param name="transb">	The transb. </param>
/// <param name="m">	 	An int to process. </param>
/// <param name="n">	 	An int to process. </param>
/// <param name="k">	 	An int to process. </param>
/// <param name="alpha"> 	The alpha. </param>
/// <param name="A">	 	A T to process. </param>
/// <param name="lda">   	The lda. </param>
/// <param name="B">	 	A T to process. </param>
/// <param name="ldb">   	The ldb. </param>
/// <param name="beta">  	The beta. </param>
/// <param name="C">	 	[in,out] If non-null, a T to process. </param>
/// <param name="ldc">   	The ldc. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
inline void devGEMM(cublasHandle_t handle,
                    cublasOperation_t transa, cublasOperation_t transb,
                    int m, int n, int k,
                    const T *alpha,
                    const T *A, int lda,
                    const T *B, int ldb,
                    const T *beta,
                    T *C, int ldc);


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Filling memory. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="A">   	[in,out] If non-null,  pointer to the array to initialize. </param>
/// <param name="n">   number of elements in the array. </param>
/// <param name="maxi">	The maxi. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> void fill(T *A, int n, int maxi) {
  for (int j = 0; j < n; j++) {
      if (std::is_same<T, float>::value || std::is_same<T, double>::value)
          A[j] = T((rand() % (maxi * 2 + 1)) - maxi) / T(maxi + 1.);
      else if (std::is_same<T, half>::value)
          A[j] = __float2half(float((rand() % (maxi * 2 + 1)) - maxi) / (maxi + 1.));
      else
          safe_exit(-1);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads a matrix. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="A">	   	[in,out] If non-null, pointer to matrix A. </param>
/// <param name="B">	   	[in,out] If non-null, pointer to matrix B. </param>
/// <param name="C">	   	[in,out] If non-null, pointer to matrix C. </param>
/// <param name="n">	   	An int to process. </param>
/// <param name="filename">	Filename of the file. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> void readMatrix(T *A, T *B, T *C, int n, string filename) {
  std::ifstream mfs(filename.c_str());
  string line;
  // Ignore header line because it was already checked
  getline(mfs, line);
  float a, b, c;
  for (int j = 0; j < n; j++) {
    sscanf(line.c_str(), "%f %f %f", &a, &b, &c);
    A[j] = T(a);
    B[j] = T(b);
    C[j] = T(c);
  }
}

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing.  The user is allowed to specify
//   the size of the input data in kiB.
//
// Arguments:
//   op: the options parser / parameter database
//
// Programmer: Anthony Danalis
// Creation: September 08, 2009
// Returns:  nothing
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op) {}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   This benchmark measures the performance of the single precision general
//   matrix multiplication (SGEMM) operation in GFLOPS.  Data transfer time
//   over the PCIe bus is not included in this measurement.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: September 08, 2009
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
   cout << "Running GEMM" << endl;
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  srand(SEED);

  bool quiet = op.getOptionBool("quiet");

  if(!quiet) {
    cout << "Running single precision test" << endl;
  }
  RunTest<float>("SGEMM", resultDB, op);


  // Test to see if this device supports double precision
  if ((deviceProp.major == 1 && deviceProp.minor >= 3) ||
      (deviceProp.major >= 2)) {
    if(!quiet) {
        cout << "Running double precision test" << endl;
    }
    RunTest<double>("DGEMM", resultDB, op);
  }

  if ((deviceProp.major >= 6)) {
    if (!quiet) {
        cout << "Running half preicsion test" << endl;
    }
    RunTest<half>("HGEMM", resultDB, op);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the test operation. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="testName">	Name of the test. </param>
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op) {
  int passes = op.getOptionInt("passes");
  int device = op.getOptionInt("device");
  const bool uvm = op.getOptionBool("uvm");
  const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
  const bool uvm_advise = op.getOptionBool("uvm-advise");
  const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
  int kib;

  // Use preset problem size or read data from input file
  string filename = op.getOptionString("inputFile");
  if (filename == "") {
    int probSizes[5] = {1, 3, 20, 60, 120};
    kib = probSizes[op.getOptionInt("size") - 1];
  } else {
    std::ifstream mfs(filename.c_str());
    std::string line;
    char object[FIELD_LENGTH];
    sscanf(line.c_str(), "%s %d", object, &kib);
  }

  // Dimensions of matrix
  int N = kib * 1024 / sizeof(T);

  // Initialize the cublas library
  cublasHandle_t handle; // CUBLAS context
  cublasStatus_t stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS initialization failed" << std::endl;
        safe_exit(-1);
  }

  // Allocate GPU memory
  T *dA, *dB, *dC;
  T *A;
  T *B;
  T *C;
  if (uvm || uvm_prefetch || uvm_advise || uvm_prefetch_advise) {
      checkCudaErrors(cudaMallocManaged(&dA, N * N* sizeof(T)));
      checkCudaErrors(cudaMallocManaged(&dB, N * N* sizeof(T)));
      checkCudaErrors(cudaMallocManaged(&dC, N * N* sizeof(T)));

      if (filename == "") {
          fill<T>(dA, N * N, 31);
          fill<T>(dB, N * N, 31);
          fill<T>(dC, N * N, 31);
      } else {
          readMatrix(dA, dB, dC, N * N, filename);
      }
  }
  else {
      checkCudaErrors(cudaMalloc(&dA, N * N * sizeof(T)));
      checkCudaErrors(cudaMalloc(&dB, N * N * sizeof(T)));
      checkCudaErrors(cudaMalloc(&dC, N * N * sizeof(T)));

      checkCudaErrors(cudaMallocHost(&A, N * N * sizeof(T)));
      checkCudaErrors(cudaMallocHost(&B, N * N * sizeof(T)));
      checkCudaErrors(cudaMallocHost(&C, N * N * sizeof(T)));

      // Fill matrix or read from input file
      if (filename == "") {
          fill<T>(A, N * N, 31);
          fill<T>(B, N * N, 31);
          fill<T>(C, N * N, 31);
      } else {
        readMatrix(A, B, C, N * N, filename);
      }
  }

  // Copy input to GPU
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  float elapsedTime;

  // Copy inputs to GPU

  double transferTime = 0;
  checkCudaErrors(cudaEventRecord(start, 0));

  if (uvm) {
      // Do nothing
  } else if (uvm_prefetch) {
      // could ignore this to test demand paging performance affect
      checkCudaErrors(cudaMemPrefetchAsync(dA, N * N * sizeof(T), device));
      cudaStream_t s1;
      checkCudaErrors(cudaStreamCreate(&s1));
      checkCudaErrors(cudaMemPrefetchAsync(dB, N * N * sizeof(T), device, s1));
      checkCudaErrors(cudaStreamDestroy(s1));
      // checkCudaErrors(cudaStreamSynchronize(0));
      // checkCudaErrors(cudaStreamSynchronize((cudaStream_t)1));
  } else if (uvm_advise) {
      // Do nothing for demand paging
      checkCudaErrors(cudaMemAdvise(dA, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, device));
      checkCudaErrors(cudaMemAdvise(dB, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, device));
  } else if (uvm_prefetch_advise) {
      checkCudaErrors(cudaMemAdvise(dA, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, device));
      checkCudaErrors(cudaMemAdvise(dB, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, device));
      checkCudaErrors(cudaMemPrefetchAsync(dA, N * N * sizeof(T), device));
      cudaStream_t s1;
      checkCudaErrors(cudaStreamCreate(&s1));
      checkCudaErrors(cudaMemPrefetchAsync(dB, N * N * sizeof(T), device, s1));
      checkCudaErrors(cudaStreamDestroy(s1));
  } else {
      checkCudaErrors(cudaMemcpy(dA, A, N * N * sizeof(T), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dB, B, N * N * sizeof(T), cudaMemcpyHostToDevice));
  }

  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3;

  bool first = true;
/// <summary>	. </summary>
  for (int j = 0; j < passes; j++) {
    for (int i = 0; i < 2; i++) {
      const cublasOperation_t transa = CUBLAS_OP_N;
      const cublasOperation_t transb = i ? CUBLAS_OP_T : CUBLAS_OP_N;
      const int nb = 128;
      const int idim = N / nb;

      int dim = idim * nb;

      const int m = dim;
      const int n = dim;
      const int k = dim;
      const int lda = dim;
      const int ldb = dim;
      const int ldc = dim;
      const T alpha = 1;
      const T beta = 0; //-1;

      // Warm Up
      devGEMM<T>(handle, transa, transb, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC,
                    ldc);
      cudaDeviceSynchronize();
      CHECK_CUDA_ERROR();

      double cublasTime;
      float kernelTime = 0.0f;
      for (int ii = 0; ii < 4; ++ii) {
          checkCudaErrors(cudaEventRecord(start, 0));
          devGEMM<T>(handle, transa, transb, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC,
                    ldc);
          checkCudaErrors(cudaEventRecord(stop, 0));
          checkCudaErrors(cudaEventSynchronize(stop));
          CHECK_CUDA_ERROR();
          float currTime = 0.0f;
          checkCudaErrors(cudaEventElapsedTime(&currTime, start, stop));
          kernelTime += currTime;
      }
      cublasTime = (kernelTime / 4.0) * 1.e-3;

      checkCudaErrors(cudaEventRecord(start, 0));    // timing may be affected by async

      if (uvm) {
        // Do nothing
      } else if (uvm_prefetch) {
          checkCudaErrors(cudaMemPrefetchAsync(dC, N * N * sizeof(T), cudaCpuDeviceId));
          // checkCudaErrors(cudaStreamSynchronize(0));
      } else if (uvm_advise) {
          checkCudaErrors(cudaMemAdvise(dC, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
          checkCudaErrors(cudaMemAdvise(dC, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
      } else if (uvm_prefetch_advise) {
          checkCudaErrors(cudaMemAdvise(dC, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
          checkCudaErrors(cudaMemAdvise(dC, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
          checkCudaErrors(cudaMemPrefetchAsync(dC, N * N * sizeof(T), cudaCpuDeviceId));
      } else {
          checkCudaErrors(cudaMemcpy(C, dC, N * N * sizeof(T), cudaMemcpyDeviceToHost));
      }

      checkCudaErrors(cudaEventRecord(stop, 0));
      checkCudaErrors(cudaEventSynchronize(stop));
      float oTransferTime = 0.0f;
      checkCudaErrors(cudaEventElapsedTime(&oTransferTime, start, stop));
      oTransferTime *= 1.e-3;

      // Add the PCIe transfer time to total transfer time only once
      if (first) {
        transferTime += oTransferTime;
        first = false;
      }

      double cublasGflops = 2. * m * n * k / cublasTime / 1e9;
      double pcieGflops = 2. * m * n * k / (cublasTime + transferTime) / 1e9;
      std::string transb_string = (transb == CUBLAS_OP_T)? "T" : "N";
      string atts = "dim:" + toString(dim);
      resultDB.AddResult(testName + "-" + transb_string + "-TransferTime", atts, "sec", transferTime);
      resultDB.AddResult(testName + "-" + transb_string + "-KernelTime", atts, "sec", cublasTime);
      resultDB.AddResult(testName + "-" + transb_string + "-TotalTime", atts, "sec", transferTime + cublasTime);
      resultDB.AddResult(testName + "-" + transb_string, atts, "GFlops", cublasGflops);
      resultDB.AddResult(testName + "-" + transb_string + "_PCIe", atts, "GFlops", pcieGflops);
      resultDB.AddResult(testName + "-" + transb_string + "_Parity", atts, "N", transferTime / cublasTime);
      resultDB.AddOverall("GFlops", "", cublasGflops);
    }
  }

  // Clean Up

  checkCudaErrors(cudaFree(dA));
  checkCudaErrors(cudaFree(dB));
  checkCudaErrors(cudaFree(dC));
  if (!uvm && !uvm_prefetch && !uvm_advise && !uvm_prefetch_advise) {
    checkCudaErrors(cudaFreeHost(A));
    checkCudaErrors(cudaFreeHost(B));
    checkCudaErrors(cudaFreeHost(C));
  }

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  cublasDestroy(handle);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   gemm kernel (double). </summary>
///
/// <typeparam name="double">	Type of the double. </typeparam>
/// <param name="transa">	The transa. </param>
/// <param name="transb">	The transb. </param>
/// <param name="m">	 	An int to process. </param>
/// <param name="n">	 	An int to process. </param>
/// <param name="k">	 	An int to process. </param>
/// <param name="alpha"> 	The alpha. </param>
/// <param name="A">	 	A double to process. </param>
/// <param name="lda">   	The lda. </param>
/// <param name="B">	 	A double to process. </param>
/// <param name="ldb">   	The ldb. </param>
/// <param name="beta">  	The beta. </param>
/// <param name="C">	 	[in,out] If non-null, a double to process. </param>
/// <param name="ldc">   	The ldc. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline void devGEMM<double>(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const double *alpha,
                            const double *A, int lda,
                            const double *B, int ldb,
                            const double *beta,
                            double *C, int ldc) {
  cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	gemm kernel (float). </summary>
///
/// <typeparam name="float">	Type of the float. </typeparam>
/// <param name="transa">	The transa. </param>
/// <param name="transb">	The transb. </param>
/// <param name="m">	 	An int to process. </param>
/// <param name="n">	 	An int to process. </param>
/// <param name="k">	 	An int to process. </param>
/// <param name="alpha"> 	The alpha. </param>
/// <param name="A">	 	A float to process. </param>
/// <param name="lda">   	The lda. </param>
/// <param name="B">	 	A float to process. </param>
/// <param name="ldb">   	The ldb. </param>
/// <param name="beta">  	The beta. </param>
/// <param name="C">	 	[in,out] If non-null, a float to process. </param>
/// <param name="ldc">   	The ldc. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline void devGEMM<float>(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float *alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           const float *beta,
                           float *C, int ldc) {
  cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline void devGEMM<half>(cublasHandle_t handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k,
                          const half *alpha,
                          const half *A, int lda,
                          const half *B, int ldb,
                          const half *beta,
                          half *C, int ldc) {
  cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

