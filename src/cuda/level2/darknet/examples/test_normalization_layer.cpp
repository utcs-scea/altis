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
#define SIZE        5
#define ALPHA       0.5
#define BETA        0.5
#define KAPPA       0.5

using namespace std;

void test_normalization_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    cout << "Begin testing normalization layer..." << endl;
    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        test_normalization_layer_forward(BATCH, WIDTH, HEIGHT, CHANNEL, SIZE, ALPHA,
                        BETA, KAPPA);
        test_normalization_layer_backward(BATCH, WIDTH, HEIGHT, CHANNEL, SIZE, ALPHA,
                        BETA, KAPPA);
    } else if (is_bidirectional == 0) {
        test_normalization_layer_forward(BATCH, WIDTH, HEIGHT, CHANNEL, SIZE, ALPHA,
                        BETA, KAPPA);
    } else if (is_bidirectional == -1) {
        test_normalization_layer_backward(BATCH, WIDTH, HEIGHT, CHANNEL, SIZE, ALPHA,
                        BETA, KAPPA);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
