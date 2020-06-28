#ifdef _cplusplus
extern "C" {
#endif

#include "darknet.h"

#ifdef _cplusplus
}
#endif

#include "OptionParser.h"
#include "ResultDatabase.h"
#include <iostream>
#include <string>

using namespace std;

void test_avgpool_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    cout << "Begin test avgpool layer..." << endl;

    int size = op.getOptionInt("size") - 1;
    int imgDims[4] = {56, 112, 224, 448};
    int batchSizes[4] = {32, 64, 128, 256};

    int imgDim = imgDims[size];
    int batchSize = batchSizes[size];
    int channel = 3;    // rgb, user can specify TODO

    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        test_avgpool_layer_forward(batchSize, imgDim, imgDim, channel);
        test_avgpool_layer_backward(batchSize, imgDim, imgDim, channel);
    } else if (is_bidirectional == 0) {
        test_avgpool_layer_forward(batchSize, imgDim, imgDim, channel);
    } else if (is_bidirectional == -1) {
        test_avgpool_layer_backward(batchSize, imgDim, imgDim, channel);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
