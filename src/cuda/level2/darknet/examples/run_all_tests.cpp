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
extern void test_activation_layer(ResultDatabase &DB, OptionParser &op);
extern void test_avgpool_layer(ResultDatabase &DB, OptionParser &op);
extern void test_batchnorm_layer(ResultDatabase &DB, OptionParser &op);
extern void test_convolutional_layer(ResultDatabase &DB, OptionParser &op); 
extern void test_crnn_layer(ResultDatabase &DB, OptionParser &op);
extern void test_deconvolutional_layer(ResultDatabase &DB, OptionParser &op);
extern void test_dropout_layer(ResultDatabase &DB, OptionParser &op);
extern void test_l2norm_layer(ResultDatabase &DB, OptionParser &op);
extern void test_logistic_layer(ResultDatabase &DB, OptionParser &op);
extern void test_maxpool_layer(ResultDatabase &DB, OptionParser &op);
extern void test_normalization_layer(ResultDatabase &DB, OptionParser &op);
extern void test_shortcut_layer(ResultDatabase &DB, OptionParser &op);
extern void test_softmax_layer(ResultDatabase &DB, OptionParser &op);

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
        test_activation_layer(DB, op);
        test_avgpool_layer(DB, op);
        test_batchnorm_layer(DB, op);
        test_convolutional_layer(DB, op);
        //test_crnn_layer(DB, op);
        test_deconvolutional_layer(DB, op);
        test_dropout_layer(DB, op);
        test_l2norm_layer(DB, op);
        test_logistic_layer(DB, op);
        test_maxpool_layer(DB, op);
        test_normalization_layer(DB, op);
        test_shortcut_layer(DB, op);
        test_softmax_layer(DB, op);
    } else if (test_type.compare("connected") == 0) {
        test_connected_layer(DB, op);
    } else if (test_type.compare("activation") == 0) {
        test_activation_layer(DB, op);
    } else if (test_type.compare("avgpool") == 0) {
        test_avgpool_layer(DB, op);
    } else if (test_type.compare("batchnorm") == 0) {
        test_batchnorm_layer(DB, op);
    } else if (test_type.compare("convolution") == 0) {
        test_convolutional_layer(DB, op);
    } else if (test_type.compare("crnn") == 0) {
        test_crnn_layer(DB, op);
    } else if (test_type.compare("deconvolution") == 0) {
        test_deconvolutional_layer(DB, op);
    } else if (test_type.compare("dropout") == 0) {
        test_dropout_layer(DB, op);
    } else if (test_type.compare("l2norm") == 0) {
        test_l2norm_layer(DB, op);
    } else if (test_type.compare("logistic") == 0) {
        test_logistic_layer(DB, op);
    } else if (test_type.compare("maxpool") == 0) {
        test_maxpool_layer(DB, op);
    } else if (test_type.compare("normalization") == 0) {
        test_normalization_layer(DB, op);
    } else if (test_type.compare("shortcut") == 0) {
        test_shortcut_layer(DB, op);
    } else if (test_type.compare("softmax") == 0) {
        test_softmax_layer(DB, op);
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

