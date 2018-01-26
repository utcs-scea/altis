# === Basics ===
#CC       = gcc
#CXX      = g++
#LD       = g++
#AR       = ar
#RANLIB   = ranlib

# In CPPFLAGS, note src/common is from the Mirovia source tree, so we must
# use $(srcdir).  In contrast, the files in config used in the build are
# generated as part of the configuration, so we want to find them in the
# build tree - hence we do not use $(srcdir) for that -I specification.
#CPPFLAGS += -I$(top_srcdir)/src/common -I$(top_builddir)/config 
CPPFLAGS += -I$(top_srcdir)/src/common -I$(top_builddir)/config
#CFLAGS   += -g -O2
#CXXFLAGS += -g -O2
NVCXXFLAGS = -g -O2
#ARFLAGS  = rcv
#LDFLAGS  =  -L$(top_builddir)/src/common
LDFLAGS  += -L$(top_builddir)/src/common
LIBS     =  -lrt

USE_MPI         = @USE_MPI@
MPICXX          = @MPICXX@
MPI_CPPFLAGS	= -DPARALLEL

OCL_CPPFLAGS    = -I$(top_srcdir)/src/opencl/common
OCL_LDFLAGS		= -L$(top_builddir)/src/opencl/common
OCL_LIBS        = -lMiroviaCommonOpenCL -lMiroviaCommon 

NVCC            = /usr/local/cuda-8.0/bin/nvcc
CUDA_CXX        = /usr/local/cuda-8.0/bin/nvcc
CUDA_INC        = -I/usr/local/cuda-8.0/include -I$(top_srcdir)/src/cuda/common
CUDA_LDFLAGS	= -L$(top_builddir)/src/cuda/common
CUDA_CPPFLAGS   = -gencode=arch=compute_60,code=sm_60 -I$(top_srcdir)/src/cuda/common

USE_CUDA        = yes
ifeq ($(USE_CUDA),yes)
CUDA_LIBS		:= -lMiroviaCommon $(shell $(top_srcdir)/config/find_cuda_libs.sh /usr/local/cuda-8.0/bin/nvcc)
else
CUDA_LIBS       =
endif



