include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC)

# What is the destination for programs built from this directory?
cudadir = $(bindir)/CUDA

# What programs should be installed to that destination?
cuda_PROGRAMS = lavaMD

# How to build those programs?
lavaMD_SOURCES = \
lavaMD.cpp \
lavaMD.h \
util/device/device.cu \
util/device/device.h \
kernel/kernel_gpu_cuda_wrapper.cu \
kernel/kernel_gpu_cuda_wrapper.h \
main.cpp \
util/timer/timer.c

lavaMD_LDADD = \
device.o \
kernel_gpu_cuda_wrapper.o \
$(CUDA_LIBS) $(LIBS) -lm

