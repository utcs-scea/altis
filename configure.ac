# Use this file as input to autoconf to output a configure script.

AC_PREREQ([2.65])
AC_INIT([ALTIS], [1.0.0], [bodunhu@utexas.com])
AC_CONFIG_AUX_DIR([build-aux])
AC_CANONICAL_SYSTEM
AC_CONFIG_SRCDIR([src/common/Timer.h])
AC_CONFIG_HEADERS([config/config.h])
AM_MAINTAINER_MODE([disable])
AM_INIT_AUTOMAKE([foreign subdir-objects])

# Altis is a collection of C++ programs AC_LANG([C++])
AC_LANG([C++])

# Set default installation location to the build location
AC_PREFIX_DEFAULT(`pwd`)

# Check for fundamental programs.
# TODO
AC_PROG_CXX
AC_PROG_CC
AC_PROG_RANLIB
AC_CHECK_TOOL([AR], [ar], [:])
AC_PATH_PROG([UNZIP], [unzip])

# Check whether special NVCXXFLAGS were specified
# If not, set default flags based on whether we're using g++ and whether our
# C++ compiler supports -g.
# Logic based on the logic used in the AC_PROG_CXX macro.
ac_save_NVCXXFLAGS=${NVCXXFLAGS}
ac_test_NVCXXFLAGS=${NVCXXFLAGS+set}
if test "$ac_test_NVCXXFLAGS" = set
then
    NVCXXFLAGS=$ac_save_NVCXXFLAGS
elif test $ac_cv_prog_cxx_g = yes
then
    if test "$GXX" = yes; then
        NVCXXFLAGS="-g -O2"  #-O2
    else
        NVCXXFLAGS="-g"
    fi
else
    if test "$GXX" = yes
    then
        NVCXXFLAGS="-O2" #-O2
    else
        NVCXXFLAGS=
    fi
fi

# On Linux, clock_gettime function seems to require the rt library
AS_CASE([$host],
    [*-*-linux*], [LIBS="$LIBS -lrt"],
    [*], []
)

# Check for header files.
AC_FUNC_ALLOCA
AC_CHECK_HEADERS([float.h stdlib.h string.h sys/time.h sys/timeb.h time.h unistd.h])

# Check for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
AC_C_RESTRICT
AC_TYPE_SIZE_T

# Check for library functions.
AC_FUNC_ERROR_AT_LINE
AC_CHECK_FUNCS([clock_gettime floor gethostname gettimeofday localtime_r memset pow sqrt strdup])

AS_IF([test "x$ac_cv_func_clock_gettime" = xyes],
    [ AC_MSG_CHECKING([whether clock_gettime() supports CLOCK_PROCESS_CPUTIME_ID timer])
      AC_LINK_IFELSE([AC_LANG_PROGRAM([],[])],
                    [AC_MSG_RESULT([yes])
                     AC_DEFINE([HAVE_CLOCK_PROCESS_CPUTIME_ID],[1],[Define to 1 if you have the CLOCK_PROCESS_CPUTIME_ID timer for the 'clock_gettime' function])
                    ],
                    [AC_MSG_RESULT([no])]
      )
    ]
)    

# CUDA support
AC_ARG_WITH([cuda],
    [AS_HELP_STRING([--with-cuda],
        [build CUDA versions of ALTIS benchmark programs @<:@default=check@:>@])],
    [],
    [with_cuda=check])

USE_CUDA=no
AS_IF([test "x$with_cuda" != xno],
    [
        # Check for the nvcc program
        AC_PATH_PROG([NVCC],[nvcc],[])

        AS_IF([test "x$NVCC" != x], [ savedCXX=$CXX
              CXX=$NVCC
              savedLIBS=$LIBS
              savedCPPFLAGS=$CPPFLAGS
              savedCXXFLAGS=$CXXFLAGS
              savedLDFLAGS=$LDFLAGS
              CXXFLAGS=$NVCXXFLAGS
              LDFLAGS=""

              # Find cuda include path based on location of nvcc command
              cudabindir=`AS_DIRNAME([$NVCC])`
              CUDA_INCDIR=`AS_DIRNAME([$cudabindir])`/include
              CPPFLAGS="$CPPFLAGS -I$CUDA_INCDIR"

              # OPTIX_HDR_FILE=/usr/local/NVIDIA-OptiX-SDK-6.5.0-linux64/include/optix.h
              OPTIX_HDR_FILE=$OPTIX_INSTALL_PATH/include/optix.h
              OPTIX_PATH=`AS_DIRNAME[$OPTIX_HDR_FILE]`
              AC_CHECK_FILE([$OPTIX_HDR_FILE],
                  [],
                  [NVCC=""],
              )

              AC_CHECK_HEADERS([cuda.h cuda_runtime.h],[],
                  [NVCC=""]
                  )
              AC_CHECK_LIB([cublas], [cublasInit],[],
                  [NVCC=""]
                  )
              AC_CHECK_LIB([cufft], [cufftPlan1d], [],
                  [NVCC=""]
                  )
              AC_CHECK_LIB([cudnn], [cudnnGetRNNDescriptor], [],
                  [NVCC=""]
                  )

              CPPFLAGS=$savedCPPFLAGS
              LIBS=$savedLIBS
              CXX=$savedCXX
              CXXFLAGS=$savedCXXFLAGS LDFLAGS=$savedLDFLAGS ],
            [ if test "x$with_cuda" != xcheck; then
                AC_MSG_FAILURE([CUDA support was requested, but no usable CUDA installation found])
              fi
              NVCC=""
            ]
        )

        AS_IF([test "x$NVCC" != x],
            [with_cuda=yes
             USE_CUDA=yes
             AC_SUBST([CUDA_INCDIR])],
            [AC_MSG_NOTICE([no usable CUDA installation found])
             with_cuda=no
            ])

        AS_IF([test "x$with_cuda" = xyes],
            [AS_IF([test "x$CUDA_CPPFLAGS" = x],
                [dnl generate CUDA code for broad spectrum of devices (Pascal arch specific)
                 CUDA_CPPFLAGS=["-arch=sm_61 -gencode=arch=compute_61,code=sm_61, --cudart shared"]
                 AC_MSG_NOTICE([using CUDA_CPPFLAGS=$CUDA_CPPFLAGS])
                 AC_MSG_WARN([*** ALTIS programs may fail if these -gencode values do not apply to your GPU.  If they don't apply, or if you want to shorten ALTIS's compile time, determine the -gencode options that apply for your GPU and re-configure ALTIS explicitly specifying only those options in CUDA_CPPFLAGS during the configure. ***])
                ]
             )
             AC_SUBST([CUDA_CPPFLAGS])
            ]
        )
    ]
)
AM_CONDITIONAL([BUILD_CUDA], [test "x$with_cuda" = "xyes"])


# test whether or not to build new cuda features in benchmarks
check_cuda_version()
{
    nvcc_version_string=$(nvcc --version | grep "Cuda compilation tools, release")    
    if echo "$nvcc_version_string" | grep -q "10"
    then
        return 10
    elif echo "$nvcc_version_string" | grep -q "9"
    then
        return 9
    elif echo "$nvcc_version_string" | grep -q "8"
    then
        return 8
    elif echo "$nvcc_version_string" | grep -q "7"
    then
        return 7
    elif echo "$nvcc_version_string" | grep -q "6"
    then
        return 6
    elif echo "$nvcc_version_string" | grep -q "5"
    then
        return 5
    elif echo "$nvcc_version_string" | grep -q "4"
    then
        return 4
    elif echo "$nvcc_version_string" | grep -q "3"
    then
        return 3
    elif echo "$nvcc_version_string" | grep -q "2"
    then
        return 2
    elif echo "$nvcc_version_string" | grep -q "1"
    then
        return 1
    else
        echo "--------ERROR: nvcc not found---------"
        exit
    fi
}

# TODO: might need to enable more features since CUDA 10
unified_memory=yes
grid_sync=yes
hyperq=yes
dynamic_parallelism=yes
task_graph=yes

check_cuda_version
cuda_version=$?

AS_IF([test "x$cuda_version" = x9], 
    [
        task_graph=no
    ])
AS_IF([test "x$cuda_version" = x8], 
    [
        grid_sync=no
        task_graph=no
    ])
AS_IF([test "x$cuda_version" = x7], 
    [
        grid_sync=no
        task_graph=no    
    ])
AS_IF([test "x$cuda_version" = x6], 
    [
        grid_sync=no
        task_graph=no
    ])
AS_IF([test "x$cuda_version" = x5], 
    [
        grid_sync=no
        unified_memory=no
        task_graph=no
    ])
AS_IF([test "x$cuda_version" = x4],
    [
        grid_sync=no
        unified_memory=no
        hyperq=no
        dynamic_parallelism=no
        task_graph=no
    ])
AS_IF([test "x$cuda_version" = x3],
    [
        grid_sync=no
        unified_memory=no
        hyperq=no
        dynamic_parallelism=no
        task_graph=no
    ])
AS_IF([test "x$cuda_version" = x2], 
    [
        grid_sync=no
        unified_memory=no
        hyperq=no
        dynamic_parallelism=no
        task_graph=no
    ])
AS_IF([test "x$cuda_version" = x1], 
    [
        grid_sync=no
        unified_memory=no
        hyperq=no
        dynamic_parallelism=no
        task_graph=no
    ])


AS_IF([test "x$task_graph" = "xyes"], 
    [
       AC_MSG_RESULT([supports cuda graph: yes])
       CUDA_CPPFLAGS+=[" -DTASK_GRAPH"]
    ],
    [
       AC_MSG_RESULT([supports cuda graph: no])
    ])
AS_IF([test "x$unified_memory" = "xyes"], 
    [
       AC_MSG_RESULT([supports unified memory: yes])
       CUDA_CPPFLAGS+=[" -DUNIFIED_MEMORY"]
    ],
    [
       AC_MSG_RESULT([supports unified memory: no])
    ])

AS_IF([test "x$grid_sync" = "xyes"], 
    [
        AC_MSG_RESULT([supports grid sync: yes])
        CUDA_CPPFLAGS+=[" -DGRID_SYNC"]
    ],
    [
        AC_MSG_RESULT([supports grid sync: no])
    ])

AS_IF([test "x$hyperq" = "xyes"], 
    [
        AC_MSG_RESULT([supports hyperq: yes])
        CUDA_CPPFLAGS+=[" -DHYPERQ"]
    ],
    [
        AC_MSG_RESULT([supports hyperq: no])
    ])

AS_IF([test "x$dynamic_parallelism" = "xyes"], 
    [
        AC_MSG_RESULT([supports dynamic parallelism: yes])
        CUDA_CPPFLAGS+=[" -DDYNAMIC_PARALLELISM"]
    ],
    [
        AC_MSG_RESULT([supports dynamic parallelism: no])
    ])


AC_SUBST([USE_CUDA])
AC_SUBST([NVCXXFLAGS])

AC_CONFIG_FILES([Makefile
                 config/common.mk
                 config/config.mk
                 config/dirtargets.mk
                 config/targets.mk
                 src/Makefile
                 src/common/Makefile
                 src/cuda/Makefile
                 src/cuda/level0/Makefile
                 src/cuda/level0/busspeeddownload/Makefile
                 src/cuda/level0/busspeedreadback/Makefile
                 src/cuda/level0/devicememory/Makefile
                 src/cuda/level0/maxflops/Makefile
                 src/cuda/level1/Makefile
                 src/cuda/level1/bfs/Makefile
                 src/cuda/level1/gemm/Makefile
                 src/cuda/level1/pathfinder/Makefile
                 src/cuda/level1/sort/Makefile
                 src/cuda/level2/Makefile
                 src/cuda/level2/cfd/Makefile
                 src/cuda/level2/dwt2d/Makefile
                 src/cuda/level2/gups/Makefile
                 src/cuda/level2/kmeans/Makefile
                 src/cuda/level2/lavamd/Makefile
                 src/cuda/level2/mandelbrot/Makefile
                 src/cuda/level2/darknet/Makefile
                 src/cuda/level2/nw/Makefile
                 src/cuda/level2/particlefilter/Makefile
                 src/cuda/level2/raytracing/Makefile
                 src/cuda/level2/raytracing/raytracing_cuda/Makefile
                 src/cuda/level2/raytracing/raytracing_optix/Makefile
                 src/cuda/level2/srad/Makefile
                 src/cuda/level2/where/Makefile
                 ])
AC_OUTPUT

