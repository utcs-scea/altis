include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common:$(srcdir)/dwt_cuda

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC)

# What is the destination for programs built from this directory?
cudadir = $(bindir)/CUDA

# What programs should be installed to that destination?
cuda_PROGRAMS = dwt2d

# How to build those programs?
dwt2d_SOURCES = \
main.cpp

dwt2d_LDADD = \
main.o \
dwt.o \
components.o \
fdwt53.o \
fdwt97.o \
rdwt53.o \
rdwt97.o \
common.o \
$(CUDA_LIBS) $(LIBS)


