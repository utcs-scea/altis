include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC)

# What is the destination for programs built from this directory?
cudadir = $(bindir)/CUDA


cuda_PROGRAMS = raytracing

raytracing_SOURCES = \
main.cpp

raytracing_LDADD = \
raytracing.o \
$(CUDA_LIBS) $(LIBS) -lm
