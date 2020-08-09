include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC)

# What is the destination for programs built from this directory?
cudadir = $(bindir)/CUDA


cuda_PROGRAMS = raytracing optix_raytracing

raytracing_optix/programs/dielectric.o: raytracing_optix/programs/dielectric.cu
	$(NVCC) ${CUDA_CPPFLAGS} $(CPPFLAGS) $(NVCXXFLAGS) -m64 -ptx -c $< -o raytracing_optix/programs/dielectric.ptx

raytracing_optix/programs/lambertian.o: raytracing_optix/programs/lambertian.cu
	$(NVCC) ${CUDA_CPPFLAGS} $(CPPFLAGS) $(NVCXXFLAGS) -m64 -ptx -c $< -o raytracing_optix/programs/lambertian.ptx

raytracing_optix/programs/miss.o: raytracing_optix/programs/miss.cu
	$(NVCC) ${CUDA_CPPFLAGS} $(CPPFLAGS) $(NVCXXFLAGS) -m64 -ptx -c $< -o raytracing_optix/programs/miss.o

raytracing_optix/programs/metal.o: raytracing_optix/programs/metal.cu
	$(NVCC) ${CUDA_CPPFLAGS} $(CPPFLAGS) $(NVCXXFLAGS) -m64 -ptx -c $< -o raytracing_optix/programs/metal.ptx

raytracing_optix/programs/raygen.o: raytracing_optix/programs/raygen.cu
	$(NVCC) ${CUDA_CPPFLAGS} $(CPPFLAGS) $(NVCXXFLAGS) -m64 -ptx -c $< -o raytracing_optix/programs/raygen.ptx

raytracing_optix/programs/sphere.o: raytracing_optix/programs/sphere.cu
	$(NVCC) ${CUDA_CPPFLAGS} $(CPPFLAGS) $(NVCXXFLAGS) -m64 -ptx -c $< -o raytracing_optix/programs/sphere.ptx

raytracing_optix/raytracing_optix.o: raytracing_optix/raytracing_optix.cpp
	$(NVCC) ${CUDA_CPPFLAGS} $(CPPFLAGS) $(NVCXXFLAGS) -I./raytracing_optix/programs -c $< -o raytracing_optix/raytracing_optix.o



raytracing_SOURCES = \
main.cpp

raytracing_LDADD = \
raytracing_cuda/raytracing.o \
$(CUDA_LIBS) $(LIBS) -lm

optix_raytracing_SOURCES = \
main.cpp

optix_raytracing_LDADD = \
raytracing_optix/raytracing_optix.o \
raytracing_optix/programs/dielectric.o \
raytracing_optix/programs/lambertian.o \
raytracing_optix/programs/metal.o \
raytracing_optix/programs/miss.o \
raytracing_optix/programs/raygen.o \
raytracing_optix/programs/sphere.o \
$(CUDA_LIBS) $(LIBS) -lm
