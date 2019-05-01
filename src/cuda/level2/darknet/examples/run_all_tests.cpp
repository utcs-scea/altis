#ifdef _cplusplus
extern "C" {
#endif
#include "darknet.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef _cplusplus
}
#endif

#include "OptionParser.h"
#include "ResultDatabase.h"
#include <iostream>
#include <string>

extern void test_connected_layer(ResultDatabase &DB, OptionParser &op);

using namespace std;

void addBenchmarkSpecOptions(OptionParser &op) {
  op.addOption("test_type", OPT_STRING, "all", "specify what type of test to run", '\0');
  op.addOption("bidirection", OPT_INT, "1", "choose whether the test include both \
                        forward and backward, 1 for both, 0 for fwd, -1 for backward", '\0');
}

//int main(int argc, char **argv)
void RunBenchmark(ResultDatabase &DB, OptionParser &op)
{
    //test_box();
    //test_activation_layer_backward();
    //test_avgpool_layer_backward();
    //test_batchnorm_layer_training_forward();
    //test_dropout_layer_forward();
    //test_softmax_layer_forward();
    //test_maxpool_layer_backward();
    //test_convolutional_layer_forward();
    std::string test_type = op.getOptionString("test_type");
    if (test_type.compare("all") == 0) {
        test_connected_layer(DB, op);
    } else if (test_type.compare("connected_layer") == 0) {
        test_connected_layer(DB,op);
    }
    


    /*
    if (argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);
    if (find_arg(argc, argv, "-nogpu")) {
        gpu_index = -1;
    }

#ifndef GPU
    gpu_index = -1;
#else
    if (gpu_index >= 0){
        cuda_set_device(gpu_index);
    }
#endif

    if (0 == strcmp(argv[1], "yolo")){
        run_yolo(argc, argv);
    } else if (0 == strcmp(argv[1], "detector")) {
        run_detector(argc, argv);
    } else if (0 == strcmp(argv[1], "rnn")) {
        run_char_rnn(argc, argv);
    } else if (0 == strcmp(argv[1], "classify")){
        predict_classifier("cfg/imagenet1k.data", argv[2], argv[3], argv[4], 5);
    } else if (0 == strcmp(argv[1], "classifier")){
        run_classifier(argc, argv);
    } else if (0 == strcmp(argv[1], "nightmare")){
        run_nightmare(argc, argv);
    } else if (0 == strcmp(argv[1], "ops")){
        operations(argv[2]);
    } else if (0 == strcmp(argv[1], "speed")){
        speed(argv[2], (argc > 3 && argv[3]) ? atoi(argv[3]) : 0);
    } else if (0 == strcmp(argv[1], "print")){
        print_weights(argv[2], argv[3], atoi(argv[4]));
    } else if (0 == strcmp(argv[1], "visualize")){
        visualize(argv[2], (argc > 3) ? argv[3] : 0);
    } else if (0 == strcmp(argv[1], "mkimg")){
        mkimg(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), argv[7]);
    } else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);
    }
    */
}
