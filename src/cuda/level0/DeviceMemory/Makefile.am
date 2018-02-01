include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC)

# What is the destination for programs built from this directory?
cudadir = $(bindir)/CUDA

# What programs should be installed in the bin/CUDA destination?
cuda_PROGRAMS = DeviceMemory

DeviceMemory_SOURCES	= main.cpp
DeviceMemory_LDADD		= DeviceMemory.o $(CUDA_LIBS) $(LIBS)
