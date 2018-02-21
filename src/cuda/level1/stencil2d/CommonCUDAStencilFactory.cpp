#include "CommonCUDAStencilFactory.h"
#include "InvalidArgValue.h"
#include <cassert>
#include <iostream>
#include <string>

template <class T>
void CommonCUDAStencilFactory<T>::CheckOptions(const OptionParser &opts) const {
  // let base class check its options first
  StencilFactory<T>::CheckOptions(opts);

  // check options
  long long blockRows = opts.getOptionInt("blockRows");
  long long blockCols = opts.getOptionInt("blockCols");
  if (blockRows <= 0 || blockCols <= 0) {
    throw InvalidArgValue("All size values must be positive");
  }

  long long matrixRows = opts.getOptionInt("matrixRows");
  long long matrixCols = opts.getOptionInt("matrixCols");

  // if both are zero, we're using a non-custom size, skip this test
  if (matrixRows == 0 || matrixCols == 0) {
    return;
  }

  size_t gRows = (size_t)matrixRows;
  size_t gCols = (size_t)matrixCols;
  size_t lRows = (size_t)blockRows;
  size_t lCols = (size_t)blockCols;

  // verify that local dimensions evenly divide global dimensions
  if (((gRows % lRows) != 0) || (lRows > gRows)) {
    throw InvalidArgValue("Number of rows must be even multiple of local rows");
  }
  if (((gCols % lCols) != 0) || (lCols > gCols)) {
    throw InvalidArgValue(
        "Number of columns must be even multiple of local columns");
  }
}

template <class T>
void CommonCUDAStencilFactory<T>::ExtractOptions(
    const OptionParser &options, T &wCenter, T &wCardinal, T &wDiagonal,
    size_t &lRows, size_t &lCols, std::vector<long long> &devices) {
  // let base class extract its options
  StencilFactory<T>::ExtractOptions(options, wCenter, wCardinal, wDiagonal);

  // extract our options
  long long blockRows = options.getOptionInt("blockRows");
  long long blockCols = options.getOptionInt("blockCols");
  lRows = (size_t)blockRows;
  lCols = (size_t)blockCols;

  // determine which device to use
  // We would really prefer this to be done in main() but
  // since BuildStencil is a virtual function, we cannot change its
  // signature, and OptionParser provides no way to override an
  // option's value after it is set during parsing.
  devices = options.getOptionVecInt("device");
}
