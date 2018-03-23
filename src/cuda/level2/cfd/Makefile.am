include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC)

# What is the destination for programs built from this directory?
cudadir = $(bindir)/CUDA

# What programs should be installed to that destination?
cuda_PROGRAMS = cfd cfd_double

# How to build those programs?
cfd_SOURCES = \
main.cpp

cfd_LDADD = \
euler3d.o \
$(CUDA_LIBS) $(LIBS) -lm

cfd_double_SOURCES = \
main.cpp

cfd_double_LDADD = \
euler3d_double.o \
$(CUDA_LIBS) $(LIBS) -lm
