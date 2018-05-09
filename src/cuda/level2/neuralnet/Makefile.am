include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common:$(srcdir)/include

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC)
EXTRAFLAGS = -std=c++11

# What is the destination for programs built from this directory?
cudadir = $(bindir)/CUDA

# What programs should be installed to that destination?
cuda_PROGRAMS = convolution

# How to build those programs?
convolution_SOURCES = \
main.cpp

convolution_LDADD = \
mat_gpu.o \
cuda_util.o \
cuda_print.o \
$(CUDA_LIBS) $(LIBS) -lm


