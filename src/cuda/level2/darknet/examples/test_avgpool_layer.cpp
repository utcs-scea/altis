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
#define WIDTH       224      // VGG
#define HEIGHT      224
#define CHANNEL     64

using namespace std;

void test_avgpool_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    cout << "Begin test avgpool layer..." << endl;
    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        test_avgpool_layer_forward(BATCH, WIDTH, HEIGHT, CHANNEL);
        test_avgpool_layer_backward(BATCH, WIDTH, HEIGHT, CHANNEL);
    } else if (is_bidirectional == 0) {
        test_avgpool_layer_forward(BATCH, WIDTH, HEIGHT, CHANNEL);
    } else if (is_bidirectional == -1) {
        test_avgpool_layer_backward(BATCH, WIDTH, HEIGHT, CHANNEL);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
