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
#define INPUT       4096
#define OUTPUT      INPUT
#define STEPS       2
#define BATCHNORM   0
#define ADAM        0

using namespace std;

void test_gru_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    printf("Testing gru layer...\n");
    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        test_gru_layer_forward(BATCH, INPUT, OUTPUT, STEPS, BATCHNORM, ADAM);
        test_gru_layer_backward(BATCH, INPUT, OUTPUT, STEPS, BATCHNORM, ADAM);
    } else if (is_bidirectional == 0) {
        test_gru_layer_forward(BATCH, INPUT, OUTPUT, STEPS, BATCHNORM, ADAM);
    } else if (is_bidirectional == -1) {
        test_gru_layer_backward(BATCH, INPUT, OUTPUT, STEPS, BATCHNORM, ADAM);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
