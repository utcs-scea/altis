include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common:$(srcdir)/util/timer:$(srcdir)/kernel

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC)

# What is the destination for programs built from this directory?
cudadir = $(bindir)/CUDA

# What programs should be installed to that destination?
cuda_PROGRAMS = lavamd

# How to build those programs?
# used to be lavaMD.cpp
lavamd_SOURCES = \
main.cpp \
util/timer/timer.c

lavamd_LDADD = \
kernel_gpu_cuda_wrapper.o \
lavaMD.o \
$(CUDA_LIBS) $(LIBS) -lm

