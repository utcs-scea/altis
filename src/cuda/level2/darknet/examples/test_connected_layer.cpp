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
#define ACTIVATION_METHOD  RELU
#define BATCHNORM   1
#define ADAM        1

using namespace std;

void test_connected_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    int size = op.getOptionInt("size") - 1;
    int batches[4] = {32, 64, 128, 256};
    int batch = batches[size];
    int dataSizes[4] = {512, 1024, 2048, 4096};
    int dataSize = dataSizes[size];
    
    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        test_connected_layer_forward(batch, dataSize,
                dataSize, ACTIVATION_METHOD, BATCHNORM, ADAM);
        test_connected_layer_backward(batch, dataSize,
                dataSize, ACTIVATION_METHOD, BATCHNORM, ADAM);
    } else if (is_bidirectional == 0) {
        test_connected_layer_forward(batch, dataSize,
                dataSize, ACTIVATION_METHOD, BATCHNORM, ADAM);
    } else if (is_bidirectional == -1) {
        test_connected_layer_backward(batch, dataSize,
                dataSize, ACTIVATION_METHOD, BATCHNORM, ADAM);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
