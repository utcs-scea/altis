#The original BFS CUDA code was obtained from Pawan Harish and P. J. Narayanan at IIIT, who have given us permission to include it as part of Rodinia under Rodinia's license.

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(CUDA_LDFLAGS) -cudart shared
AM_CPPFLAGS = $(CUDA_INC) -Wformat

# What is the destination for programs built from this directory?
cudadir = $(bindir)/CUDA

# What programs should be installed to that destination?
cuda_PROGRAMS = bfs

bfs_LDFLAGS = -pthread

# How to build those programs?
bfs_SOURCES = main.cpp
bfs_LDADD = bfs.o $(CUDA_LIBS) $(LIBS)


