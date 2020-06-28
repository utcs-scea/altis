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

#define BATCH       128
#define WIDTH       224      // VGG
#define HEIGHT      224
#define CHANNEL     64

using namespace std;

void test_batchnorm_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    cout << "Begin testing batchnorm layer..." << endl;
    int size = op.getOptionInt("size") - 1;
    int imgDims[4] = {56, 112, 224, 448};
    int imgDim = imgDims[size];
    int channels = 3;
    int batches[4] = {32, 64, 128, 256};
    int batch = batches[size];

    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        test_batchnorm_layer_forward(batch, imgDim, imgDim, channels);
        test_batchnorm_layer_backward(batch, imgDim, imgDim, channels);
    } else if (is_bidirectional == 0) {
        test_batchnorm_layer_forward(batch, imgDim, imgDim, channels);
    } else if (is_bidirectional == -1) {
        test_batchnorm_layer_backward(batch, imgDim, imgDim, channels);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
