include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common:$(srcdir)/util/timer:$(srcdir)/util/device:$(srcdir)/kernel

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
main.cpp \
util/timer/timer.c

lavaMD_LDADD = \
device.o \
kernel_gpu_cuda_wrapper.o \
$(CUDA_LIBS) $(LIBS) -lm

