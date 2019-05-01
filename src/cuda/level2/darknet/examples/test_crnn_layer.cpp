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
#define CHANNEL     3
#define HIDDEN_LAYER      24
#define OUTPUT_FILTER     36
#define STEPS       20
#define ACTIVATION_METHOD RELU
#define BATCHNORM   1


using namespace std;

void test_crnn_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    cout << "Begin test avgpool layer..." << endl;
    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        printf("this function is not working yet!\n\n");
        exit(1);
        test_crnn_layer_forward(BATCH, HEIGHT, WIDTH, CHANNEL, HIDDEN_LAYER,
            OUTPUT_FILTER, STEPS, ACTIVATION_METHOD, BATCHNORM);
    } else if (is_bidirectional == 0) {
    } else if (is_bidirectional == -1) {
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
