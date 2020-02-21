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
#define INPUT       4096 * 2
#define GROUPS      64

using namespace std;

void test_softmax_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    printf("Testing softmax layer...\n");
    int is_bidirectional = op.getOptionInt("bidirection");
    int size = op.getOptionInt("size") - 1;
    int batches[4] = {32, 64, 128, 256};
    int batch = batches[size];
    int inputSizes[4] = {512, 1024, 2048, 4096};
    int inputSize = inputSizes[size];

    if (is_bidirectional == 1) {
        test_softmax_layer_forward(batch, inputSize, GROUPS);
        test_softmax_layer_backward(batch, inputSize, GROUPS);
    } else if (is_bidirectional == 0) {
        test_softmax_layer_forward(batch, inputSize, GROUPS);
    } else if (is_bidirectional == -1) {
        test_softmax_layer_backward(batch, inputSize, GROUPS);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
