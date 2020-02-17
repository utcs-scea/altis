include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

VPATH = $(srcdir):$(srcdir)/../..common:$(srcdir)/../../../common

CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
#NVCC_DBG       = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64
GENCODE_FLAGS  = -gencode arch=compute_60,code=sm_60

SRCS = raytracing.cu
INCS = vec3.h ray.h hitable.h hitable_list.h sphere.h camera.h material.h
DEPS = ../../../common/*.h ../../common/*.h

all: cudart

cudart: cudart.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart cudart.o

cudart.o: $(SRCS) $(INCS) $(DEPS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart.o -c raytracing.cu

out.ppm: cudart
	rm -f out.ppm
	./cudart > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

profile_basic: cudart
	nvprof ./cudart > out.ppm

# use nvprof --query-metrics
profile_metrics: cudart
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./cudart > out.ppm

clean:
	rm -f cudart cudart.o out.ppm out.jpg
