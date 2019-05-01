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
#define INPUT_SIZE  4096
#define OUTPUT_SIZE 4096
#define ACTIVATION_METHOD  LEAKY 
#define BATCHNORM   1
#define ADAM        1

using namespace std;

void test_connected_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        test_connected_layer_forward(BATCH, INPUT_SIZE,
                OUTPUT_SIZE, ACTIVATION_METHOD, BATCHNORM, ADAM);
    } else if (is_bidirectional == 0) {
        //test_connected_layer_forward();
    } else if (is_bidirectional == -1) {
        // TODO
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
