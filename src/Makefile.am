if BUILD_CUDA
    MAYBE_CUDA = cuda
endif

if BUILD_OPENCL
    MAYBE_OPENCL = opencl
endif

SUBDIRS=common $(MAYBE_OPENCL) $(MAYBE_CUDA)

