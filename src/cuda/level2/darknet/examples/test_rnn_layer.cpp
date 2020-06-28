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
#define SEQLEN      1     
#define LAYERS      2

using namespace std;

void test_rnn_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    printf("Testing rnn layer...\n");
    int size = op.getOptionInt("size") - 1;
    int batches[4] = {32, 64, 128, 256};
    int batch = batches[size];
    int inputSizes[4] = {512, 1024, 2048, 4096};
    int inputSize = inputSizes[size];
    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        test_rnn_layer_forward(batch, inputSize, inputSize, SEQLEN, LAYERS);
        test_rnn_layer_backward(batch, inputSize, inputSize, SEQLEN, LAYERS);
    } else if (is_bidirectional == 0) {
        test_rnn_layer_forward(batch, inputSize, inputSize, SEQLEN, LAYERS);
    } else if (is_bidirectional == -1) {
        test_rnn_layer_backward(batch, inputSize, inputSize, SEQLEN, LAYERS);
    } else if (is_bidirectional == 0) {
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
