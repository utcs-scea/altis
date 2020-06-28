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
#define INPUT_SIZE  (BATCH * 4096)
#define ACTIVATION_METHOD  RELU

using namespace std;

static inline void show_activation_type(ACTIVATION actv) {
    printf("Activation type: ");
    switch(actv){
        case LOGISTIC:
            printf("LOGISTIC\n");
            break;
        case RELU:
            printf("RELU\n");
            break;
        case ELU:
            printf("ELU\n");
            break;
        case LEAKY:
            printf("LEAKY\n");
            break;
        case TANH:
            printf("TANH\n");
            break;
        case HARDTAN:
            printf("HARTAN\n");
            break;
        case SIGMOID:
            printf("SIGMOID\n");
            break;
        default:
            fprintf(stderr, "%s", "Not a valid activation type!\n");
            exit(1);
    }
}

void test_activation_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    cout << "Begin activation layer test..." << endl;
    show_activation_type(ACTIVATION_METHOD);
    int batchSizes[4] = {32, 64, 128, 256};
    int inputSize[4] = {256, 512, 1024, 2048};

    int is_bidirectional = op.getOptionInt("bidirection");
    int size = op.getOptionInt("size") - 1;
    int batch = batchSizes[size];
    int input_size = inputSize[size];

    if (is_bidirectional == 1) {
        test_activation_layer_forward(batch, input_size, ACTIVATION_METHOD);
        test_activation_layer_backward(batch, input_size, ACTIVATION_METHOD);
    } else if (is_bidirectional == 0) {
        test_activation_layer_forward(batch, input_size, ACTIVATION_METHOD);
    } else if (is_bidirectional == -1) {
        test_activation_layer_backward(batch, input_size, ACTIVATION_METHOD);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
