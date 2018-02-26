include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC)

# What is the destination for programs built from this directory?
cudadir = $(bindir)/CUDA

# What programs should be installed to that destination?
cuda_PROGRAMS = particlefilter

# How to build those programs?
particlefilter_SOURCES = \
main.cpp \
harness.cpp

particlefilter_LDADD = \
ex_particle_CUDA_naive_seq.o \
ex_particle_CUDA_float_seq.o \
$(CUDA_LIBS) $(LIBS) -lm


