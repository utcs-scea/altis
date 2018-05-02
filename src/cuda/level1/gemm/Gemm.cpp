#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Timer.h"
#include "Utility.h"
#include "cublas.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cudacommon.h"
#include "cuda_fp16.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define SEED 7
static const int FIELD_LENGTH = 128;

using namespace std;

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op);

template <class T>
inline void devGEMM(char transa, char transb, int m, int n, int k, T alpha,
                    const T *A, int lda, const T *B, int ldb, T beta, T *C,
                    int ldc);


// ********************************************************
// Function: fill
//
// Purpose:
//   Simple routine to initialize input array
//
// Arguments:
//   A: pointer to the array to initialize
//   n: number of elements in the array
//
// ********************************************************
template <class T> void fill(T *A, int n, int maxi) {
  for (int j = 0; j < n; j++) {
    A[j] = T((rand() % (maxi * 2 + 1)) - maxi) / (maxi + 1.);
  }
}

// ********************************************************
// Function: readMatrix
//
// Purpose:
//   Initialize input arrays from a data file
//
// Arguments:
//   A: pointer to matrix A
//   B: pointer to matrix B
//   C: pointer to matrix C
//   n: number of elements in the array
//
// ********************************************************
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
}

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op) {
  int passes = op.getOptionInt("passes");
  int kib;

  // Use preset problem size or read data from input file
  string filename = op.getOptionString("inputFile");
  if (filename == "") {
    int probSizes[4] = {1, 3, 40, 60};
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
  cublasInit();

  // Allocate GPU memory
  T *dA, *dB, *dC;
  CUDA_SAFE_CALL(cudaMalloc(&dA, N * N * sizeof(T)));
  CUDA_SAFE_CALL(cudaMalloc(&dB, N * N * sizeof(T)));
  CUDA_SAFE_CALL(cudaMalloc(&dC, N * N * sizeof(T)));

  // Initialize host memory
  T *A;
  T *B;
  T *C;

  CUDA_SAFE_CALL(cudaMallocHost(&A, N * N * sizeof(T)));
  CUDA_SAFE_CALL(cudaMallocHost(&B, N * N * sizeof(T)));
  CUDA_SAFE_CALL(cudaMallocHost(&C, N * N * sizeof(T)));

  // Fill matrix or read from input file
  if (filename == "") {
    fill<T>(A, N * N, 31);
    fill<T>(B, N * N, 31);
    fill<T>(C, N * N, 31);
  } else {
    readMatrix(A, B, C, N * N, filename);
  }

  // Copy input to GPU
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  // Copy inputs to GPU
  double transferTime = 0;
  cudaEventRecord(start, 0);
  CUDA_SAFE_CALL(cudaMemcpy(dA, A, N * N * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dB, B, N * N * sizeof(T), cudaMemcpyHostToDevice));
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3;

  bool first = true;
  for (int j = 0; j < passes; j++) {
    for (int i = 0; i < 2; i++) {
      const char transa = 'N';
      const char transb = i ? 'T' : 'N';
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
      devGEMM<T>(transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC,
                 ldc);
      cudaThreadSynchronize();
      CHECK_CUDA_ERROR();

      double cublasTime;
      float kernelTime = 0.0f;
      for (int ii = 0; ii < 4; ++ii) {
        cudaEventRecord(start, 0);
        devGEMM<T>(transa, transb, m, n, k, alpha, dA, lda, dB, ldb, beta, dC,
                   ldc);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        CHECK_CUDA_ERROR();
        float currTime = 0.0f;
        cudaEventElapsedTime(&currTime, start, stop);
        kernelTime += currTime;
      }
      cublasTime = (kernelTime / 4.0) * 1.e-3;

      cudaEventRecord(start, 0);
      CUDA_SAFE_CALL(
          cudaMemcpy(C, dC, N * N * sizeof(float), cudaMemcpyDeviceToHost));
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      float oTransferTime = 0.0f;
      cudaEventElapsedTime(&oTransferTime, start, stop);
      oTransferTime *= 1.e-3;

      // Add the PCIe transfer time to total transfer time only once
      if (first) {
        transferTime += oTransferTime;
        first = false;
      }

      double cublasGflops = 2. * m * n * k / cublasTime / 1e9;
      double pcieGflops = 2. * m * n * k / (cublasTime + transferTime) / 1e9;
      string atts = "dim:" + toString(dim);
      resultDB.AddResult(testName + "-" + transb + "-TransferTime", atts, "sec", transferTime);
      resultDB.AddResult(testName + "-" + transb + "-KernelTime", atts, "sec", cublasTime);
      resultDB.AddResult(testName + "-" + transb + "-TotalTime", atts, "sec", transferTime + cublasTime);
      resultDB.AddResult(testName + "-" + transb, atts, "GFlops", cublasGflops);
      resultDB.AddResult(testName + "-" + transb + "_PCIe", atts, "GFlops", pcieGflops);
      resultDB.AddResult(testName + "-" + transb + "_Parity", atts, "N", transferTime / cublasTime);
      resultDB.AddOverall("GFlops", "", cublasGflops);
    }
  }

  // Clean Up
  CUDA_SAFE_CALL(cudaFree(dA));
  CUDA_SAFE_CALL(cudaFree(dB));
  CUDA_SAFE_CALL(cudaFree(dC));
  CUDA_SAFE_CALL(cudaFreeHost(A));
  CUDA_SAFE_CALL(cudaFreeHost(B));
  CUDA_SAFE_CALL(cudaFreeHost(C));
  CUDA_SAFE_CALL(cudaEventDestroy(start));
  CUDA_SAFE_CALL(cudaEventDestroy(stop));
  cublasShutdown();
}


template <>
inline void devGEMM<double>(char transa, char transb, int m, int n, int k,
                            double alpha, const double *A, int lda,
                            const double *B, int ldb, double beta, double *C,
                            int ldc) {
  cublasDgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline void devGEMM<float>(char transa, char transb, int m, int n, int k,
                           float alpha, const float *A, int lda, const float *B,
                           int ldb, float beta, float *C, int ldc) {
  cublasSgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

