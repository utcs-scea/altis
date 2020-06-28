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
#define WIDTH       224      // VGG, change however you want
#define HEIGHT      224
#define CHANNEL     64
#define FILTERS     128
#define GROUP       1
#define SIZE        2        // filter size
#define STRIDE      2
#define PADDING     1
#define ACTIVATION_METHOD RELU
#define BATCHNORM   1
#define BINARY      0
#define XNOR        0
#define ADAM        0

using namespace std;

void test_convolutional_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    cout << "Begin testing convolutional layer..." << endl;

    int size = op.getOptionInt("size") - 1;
    int batches[4] = {32, 64, 128, 256};
    int batch = batches[size];
    int imgDims[4] = {28, 56, 112, 224};
    int imgDim = imgDims[size];
    int channel = 64;
    

    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        test_convolutional_layer_forward(batch, imgDim, imgDim, channel,
                FILTERS, GROUP, SIZE, STRIDE, PADDING, ACTIVATION_METHOD,
                BATCHNORM, BINARY, XNOR, ADAM);
        test_convolutional_layer_backward(batch, imgDim, imgDim, channel,
                FILTERS, GROUP, SIZE, STRIDE, PADDING, ACTIVATION_METHOD,
                BATCHNORM, BINARY, XNOR, ADAM);
    } else if (is_bidirectional == 0) {
        test_convolutional_layer_forward(batch, imgDim, imgDim, channel,
                FILTERS, GROUP, SIZE, STRIDE, PADDING, ACTIVATION_METHOD,
                BATCHNORM, BINARY, XNOR, ADAM);
    } else if (is_bidirectional == -1) {
        test_convolutional_layer_backward(batch, imgDim, imgDim, channel,
                FILTERS, GROUP, SIZE, STRIDE, PADDING, ACTIVATION_METHOD,
                BATCHNORM, BINARY, XNOR, ADAM);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
