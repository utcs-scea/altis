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

// Default parameters for connected layer
#define BATCH       128
#define INPUT_SIZE  (128 * 4096)
#define PROB        0.3

using namespace std;

void test_dropout_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    printf("Testing dropout layer...\n");
    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        test_dropout_layer_forward(BATCH, INPUT_SIZE, PROB);
        test_dropout_layer_backward(BATCH, INPUT_SIZE, PROB);
    } else if (is_bidirectional == 0) {
        test_dropout_layer_forward(BATCH, INPUT_SIZE, PROB);
    } else if (is_bidirectional == -1) {
        test_dropout_layer_backward(BATCH, INPUT_SIZE, PROB);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
