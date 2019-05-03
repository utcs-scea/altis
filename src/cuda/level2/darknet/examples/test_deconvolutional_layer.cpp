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
#define WIDTH       112      // VGG, change however you want
#define HEIGHT      112
#define CHANNEL     3
#define FILTERS     24
#define SIZE        2
#define STRIDE      1
#define PADDING     0
#define ACTIVATION_METHOD RELU
#define BATCHNORM   1
#define BINARY      0
#define ADAM        0

using namespace std;

void test_deconvolutional_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    cout << "Begin testing deconvolutional layer..." << endl;
    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        test_deconvolutional_layer_forward(BATCH, HEIGHT, WIDTH, CHANNEL,
                FILTERS, SIZE, STRIDE, PADDING, ACTIVATION_METHOD,
                BATCHNORM, ADAM);
        test_deconvolutional_layer_backward(BATCH, HEIGHT, WIDTH, CHANNEL,
                FILTERS, SIZE, STRIDE, PADDING, ACTIVATION_METHOD,
                BATCHNORM, ADAM);
    } else if (is_bidirectional == 0) {
        test_deconvolutional_layer_forward(BATCH, HEIGHT, WIDTH, CHANNEL,
                FILTERS, SIZE, STRIDE, PADDING, ACTIVATION_METHOD,
                BATCHNORM, ADAM);
    } else if (is_bidirectional == -1) {
        test_deconvolutional_layer_backward(BATCH, HEIGHT, WIDTH, CHANNEL,
                FILTERS, SIZE, STRIDE, PADDING, ACTIVATION_METHOD,
                BATCHNORM, ADAM);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
