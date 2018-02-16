#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

extern char *strcpy();
extern void exit();
extern void load(BPNN *net);
extern void bpnn_train_cuda(BPNN *net, float *eo, float *eh);

int layer_size = 0;

void backprop_face()
{
  BPNN *net;
  int i;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  
  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_cuda(net, &out_err, &hid_err);
  bpnn_free(net);
  printf("Training done\n");
}

int setup(ResultDatabase &resultDB, OptionParser &op)
{
  int seed;

  layer_size = op.getOptionInt("layerSize");

  if (layer_size % 16 != 0) {
    fprintf(stderr, "The number of input points must be divided by 16\n");
    exit(0);
  }
  
  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
