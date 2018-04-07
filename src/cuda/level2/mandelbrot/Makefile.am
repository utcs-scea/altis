include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC)

# What is the destination for programs built from this directory?
cudadir = $(bindir)/CUDA

# What programs should be installed to that destination?
cuda_PROGRAMS = mandelbrot

# How to build those programs?
mandelbrot_SOURCES = \
main.cpp

mandelbrot_LDADD = \
mandelbrot.o \
$(CUDA_LIBS) $(LIBS) -lm
