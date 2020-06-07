include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_LDFLAGS = $(CUDA_INC)

cudadir = $(bindir)/CUDA

cuda_PROGRAMS = kmeansTest

kmeansTest_SOURCES = \
kmeans_main.cu

kmeansTest_LDADD = \
kmeans_main.o \
$(CUDA_LIBS) $(LIBS) -lm
