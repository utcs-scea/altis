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

#define BATCH       64
#define WIDTH       224      // VGG
#define HEIGHT      224
#define CHANNEL     32
#define SIZE        3
#define ALPHA       0.5
#define BETA        0.5
#define KAPPA       0.5

using namespace std;

void test_normalization_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    cout << "Begin testing normalization layer..." << endl;
    int size = op.getOptionInt("size") - 1;
    int is_bidirectional = op.getOptionInt("bidirection");

    int batches[4] = {32, 64, 128, 256};
    int imgDims[4] = {56, 112, 224, 448};
    int imgDim = imgDims[size];
    int batch = batches[size];
    int channel = 3;

    if (is_bidirectional == 1) {
        test_normalization_layer_forward(batch, imgDim, imgDim, channel, SIZE, ALPHA,
                        BETA, KAPPA);
        test_normalization_layer_backward(batch, imgDim, imgDim, channel, SIZE, ALPHA,
                        BETA, KAPPA);
    } else if (is_bidirectional == 0) {
        test_normalization_layer_forward(batch, imgDim, imgDim, channel, SIZE, ALPHA,
                        BETA, KAPPA);
    } else if (is_bidirectional == -1) {
        test_normalization_layer_backward(batch, imgDim, imgDim, channel, SIZE, ALPHA,
                        BETA, KAPPA);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
