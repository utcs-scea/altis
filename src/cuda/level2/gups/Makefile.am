include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# EXTRAFLAGS = 
# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC)

# What is the destination for programs built from this directory?
cudadir = $(bindir)/CUDA

# What programs should be installed to that destination?
cuda_PROGRAMS = gups

# How to build those programs?
gups_SOURCES = \
main.cpp 

gups_LDADD = \
gups.o \
$(CUDA_LIBS) $(LIBS) -lm
