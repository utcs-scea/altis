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

    if (is_bidirectional == 1) {
        test_softmax_layer_forward(BATCH, INPUT, GROUPS);
        test_softmax_layer_backward(BATCH, INPUT, GROUPS);
    } else if (is_bidirectional == 0) {
        test_softmax_layer_forward(BATCH, INPUT, GROUPS);
    } else if (is_bidirectional == -1) {
        test_softmax_layer_backward(BATCH, INPUT, GROUPS);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
