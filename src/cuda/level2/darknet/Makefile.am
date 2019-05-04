include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

GPU=1
CUDNN=1
DEBUG=0


VPATH=$(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common:./src/:./examples

SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet
RUN=test
OBJDIR=./obj/

CC=gcc
CPP=g++
NVCC=nvcc 
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= -Iinclude/ -Isrc/
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC

#OPTS=-O0 -g

CFLAGS+=$(OPTS)

COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand

COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
CXXFLAGS=-I./../../../common
LDFLAGS+= -lcudnn

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o upsample_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o logistic_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o l2norm_layer.o yolo_layer.o iseg_layer.o
EXECOBJA=rnn.o classifier.o yolo.o detector.o nightmare.o run_all_tests.o main.o test_activation_layer.o test_connected_layer.o test_avgpool_layer.o test_batchnorm_layer.o \
		 test_convolutional_layer.o test_crnn_layer.o test_deconvolutional_layer.o  test_normalization_layer.o\
		 test_dropout_layer.o test_l2norm_layer.o test_logistic_layer.o test_maxpool_layer.o test_shortcut_layer.o#run_test.o #darknet.o

LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o avgpool_layer_kernels.o

#AM_LDFLAGS=$(CUDA_LDFLAGS) $(LDFLAGS) $(COMMON)
#AM_CPPFLAGS = $(CUDA_INC) $(COMMON)
#AM_CFLAGS = $(CFLAGS)

# Destination for programs built from this dir
#cudadir = $(bindir)/CUDA

# What programs should be installed to that destination
#cuda_PROGRAMS = darknet

#EXTRAFLAGS = $(COMMON) --compiler-options "$(CFLAGS)" -lcuda -lcudart -lcublas -lcurand -lcudnn

# How to build those programs?
#darknet_SOURCES = run_test.c \
#				  classifier.c \
#				  detector.c \
#				  nightmare.c \
#				  rnn.c \
#				  rnn_vid.c \
#				  yolo.c

#darknet_LDADD = $(OBJ) #$(CUDA_LIBS) $(LIBS)

# Overwriting default rule
EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h ../../../common/*.h

#libMiroviaCommon
ALIBMIROVIACOMMON = ../../../common/libMiroviaCommon.a

all: obj backup results $(SLIB) $(ALIB) $(EXEC)
#darknet_SOURCES = run_test.c


$(EXEC): $(EXECOBJ) $(ALIB) $(ALIBMIROVIACOMMON)
	$(CPP) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB) $(ALIBMIROVIACOMMON)

#$(RUN): $(EXECOBJ) $(ALIB)
#	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^



$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(COMMON) $(CXXFLAGS) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(OBJDIR)/*

