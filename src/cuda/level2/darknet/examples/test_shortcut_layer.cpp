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

#define BATCH       64
#define INDEX       0
#define WIDTH       224      // VGG
#define HEIGHT      224
#define CHANNEL     64
#define WIDTH2      224
#define HEIGHT2     224
#define CHAN2       64

using namespace std;

void test_shortcut_layer(ResultDatabase &resultDB, OptionParser &op) {
    // Config parameter
    cout << "Begin testing shortcut layer..." << endl;
    int is_bidirectional = op.getOptionInt("bidirection");

    if (is_bidirectional == 1) {
        test_shortcut_layer_forward(BATCH, INDEX, WIDTH, HEIGHT, CHANNEL,
                        WIDTH2, HEIGHT2, CHAN2);
        test_shortcut_layer_backward(BATCH, INDEX, WIDTH, HEIGHT, CHANNEL,
                        WIDTH2, HEIGHT2, CHAN2);
    } else if (is_bidirectional == 0) {
        test_shortcut_layer_forward(BATCH, INDEX, WIDTH, HEIGHT, CHANNEL,
                        WIDTH2, HEIGHT2, CHAN2);
    } else if (is_bidirectional == -1) {
        test_shortcut_layer_backward(BATCH, INDEX, WIDTH, HEIGHT, CHANNEL,
                        WIDTH2, HEIGHT2, CHAN2);
    } else {
        cerr << is_bidirectional << " is not a valid bidirection flag" << endl;
        exit(1);
    }
}
