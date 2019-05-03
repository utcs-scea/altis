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
#define INPUT_SIZE  (64 * 224 * 224)

using namespace std;

void test_l2norm_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    cout << "Begin testing l2norm layer..." << endl;
    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        test_l2norm_layer_forward(BATCH, INPUT_SIZE);
        test_l2norm_layer_backward(BATCH, INPUT_SIZE);
    } else if (is_bidirectional == 0) {
        test_l2norm_layer_forward(BATCH, INPUT_SIZE);
    } else if (is_bidirectional == -1) {
        test_l2norm_layer_backward(BATCH, INPUT_SIZE);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
