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
    printf("Testing gru layer...\n");
    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        test_rnn_layer_forward(BATCH, INPUT, OUTPUT, SEQLEN, LAYERS);
        //test_gru_layer_backward(BATCH, INPUT, OUTPUT, STEPS, BATCHNORM, ADAM);
    } else if (is_bidirectional == 0) {
        test_rnn_layer_forward(BATCH, INPUT, OUTPUT, SEQLEN, LAYERS);
    } else if (is_bidirectional == -1) {
        //test_gru_layer_backward(BATCH, INPUT, OUTPUT, STEPS, BATCHNORM, ADAM);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
