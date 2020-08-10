include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

OPTIX_FILES_DIR = raytracing_optix/programs

OPTIX_FILES = $(wildcard raytracing_optix/programs/*.cu)

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC) 

# What is the destination for programs built from this directory?
cudadir = $(bindir)/CUDA

cuda_PROGRAMS = raytracing optix_raytracing

$(OPTIX_FILES_DIR)/%.ptx: $(OPTIX_FILES_DIR)/%.cu
	$(NVCC) -o $@ -c $< -m64 -ptx ${CUDA_CPPFLAGS} $(CPPFLAGS) $(NVCXXFLAGS)

$(OPTIX_FILES_DIR)/%.ptx_embedded.c: $(OPTIX_FILES_DIR)/%.ptx
	$(dir $(NVCC))bin2c -c --padd 0 --type char --name embedded_$(basename $(notdir $<))_programs $< > $@

$(OPTIX_FILES_DIR)/%.ptx_embedded.o: $(OPTIX_FILES_DIR)/%.ptx_embedded.c
	$(NVCC) ${CUDA_CPPFLAGS} $(CPPFLAGS) $(NVCXXFLAGS) -c $< -o $@

raytracing_optix/raytracing_optix.o: raytracing_optix/raytracing_optix.cpp $(OPTIX_FILES:.cu=.ptx_embedded.o)
	$(NVCC) ${CUDA_CPPFLAGS} $(CPPFLAGS) $(NVCXXFLAGS) -c $< -o raytracing_optix/raytracing_optix.o

raytracing_SOURCES = \
main.cpp

raytracing_LDADD = \
raytracing_cuda/raytracing.o \
$(CUDA_LIBS) $(LIBS) -lm

optix_raytracing_SOURCES = \
main.cpp

optix_raytracing_LDADD = \
raytracing_optix/raytracing_optix.o \
$(OPTIX_FILES:.cu=.ptx_embedded.o) \
$(CUDA_LIBS) $(LIBS) -lm

optix_raytracing_LDFLAGS = \
-rdynamic
