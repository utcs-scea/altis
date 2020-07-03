include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

EXTRAFLAGS = -rdc=true -dc

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC)

# What is the destination for programs built from this directory?
cudadir = $(bindir)/CUDA

# What programs should be installed to that destination?
cuda_PROGRAMS = kmeans

kmeans_link.o: kmmain.o
	$(NVCC) ${CUDA_CPPFLAGS} $(CPPFLAGS) $(NVCXXFLAGS) -dlink $< -o kmeans_link.o

kmeansraw_link.o: kmeansraw.o
	$(NVCC) ${CUDA_CPPFLAGS} $(CPPFLAGS) $(NVCXXFLAGS) -dlink $< -o kmeansraw_link.o

# How to build those programs?
kmeans_SOURCES = \
main.cpp 

kmeans_LDADD = \
kmmain.o \
kmeans_link.o \
kmeansraw.o \
kmeansraw_link.o \
$(CUDA_LIBS) $(LIBS) -lm
