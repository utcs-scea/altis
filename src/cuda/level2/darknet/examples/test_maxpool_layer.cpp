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
#define HEIGHT      224
#define WIDTH       224
#define CHANNEL     64
#define SIZE        2
#define STRIDE      2
#define PADDING     0

using namespace std;

void test_maxpool_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    printf("Testing maxpool layer...\n");
    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        test_maxpool_layer_forward(BATCH, HEIGHT, WIDTH, CHANNEL, SIZE, STRIDE, PADDING);
        test_maxpool_layer_backward(BATCH, HEIGHT, WIDTH, CHANNEL, SIZE, STRIDE, PADDING);
    } else if (is_bidirectional == 0) {
        test_maxpool_layer_forward(BATCH, HEIGHT, WIDTH, CHANNEL, SIZE, STRIDE, PADDING);
    } else if (is_bidirectional == -1) {
        test_maxpool_layer_backward(BATCH, HEIGHT, WIDTH, CHANNEL, SIZE, STRIDE, PADDING);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
