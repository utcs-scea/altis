# Use this file as input to autoconf to output a configure script.

AC_PREREQ([2.65])
AC_INIT([MIROVIA], [1.0.0], [sarahgw@cs.utexas.edu])
AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE([-Wall -Werror foreign]) 
AC_CONFIG_SRCDIR([src/common/Timer.h])
AC_CONFIG_HEADERS([config/config.h])

# Mirovia is a collection of C++ programs
AC_LANG([C++])

# Set default installation location to the build location
AC_PREFIX_DEFAULT(`pwd`)

# Check for fundamental programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_RANLIB
AC_CHECK_TOOL([AR], [ar], [:])
AC_PATH_PROG([PERL], [perl])
AC_PATH_PROG([UNZIP], [unzip])

# Check whether special NVCXXFLAGS were specified
# If not, set default flags based on whehter we are using g++ and whether our
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
        NVCXXFLAGS="-g -O2"
    else
        NVCXXFLAGS="-g"
    fi
else
    if test "$GXX" = yes
    then
        NVCXXFLAGS="-O2"
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

# OpenCL support
AC_ARG_WITH([opencl],
    [AS_HELP_STRING([--with-opencl],
        [build OpenCL versions of SHOC benchmark programs @<:@default=check@:>@])],
    [],
    [with_opencl=check])

# Detect/verify we have a working OpenCL header and library
AS_IF([test "x$with_opencl" != xno],
    [
        # Check for usable OpenCL header
        # Note: C++ binding uses opencl.h, not just cl.h
        AC_MSG_NOTICE([checking for usable OpenCL opencl.h header])
        AC_CHECK_HEADER([OpenCL/opencl.h],
            [OPENCL_HEADER='"OpenCL/opencl.h"'],
            [AC_CHECK_HEADER([CL/opencl.h],
                [OPENCL_HEADER='"CL/opencl.h"'],
                [if test "x$with_opencl" != xcheck; then
                    AC_MSG_FAILURE([OpenCL support was requested, but no usable OpenCL header was found])
                 fi
                ]
             )
            ]
        )

        AS_IF([test "x$OPENCL_HEADER" != x],
            [
            # we have a header, check for usable OpenCL library
            # we cannot use AC_CHECK_LIB because we want to check
            # using the OS X OpenCL framework, and AC_CHECK_LIB does not
            # know how to use the framework linker syntax.
            # (we also do not want the OpenCL libs to be added to the
            # LIBS variable, and AC_CHECK_LIB does that if it succeeds)
            AC_MSG_CHECKING([for usable OpenCL library])
            savedLIBS=$LIBS
            OPENCL_LIBS="-framework OpenCL"
            LIBS="$OPENCL_LIBS $savedLIBS"
            AC_LINK_IFELSE( [AC_LANG_PROGRAM([@%:@include $OPENCL_HEADER],[clGetPlatformIDs(0,0,0);])],
                [AC_MSG_RESULT($OPENCL_LIBS)],
                [OPENCL_LIBS="-lOpenCL"
                 LIBS="$OPENCL_LIBS $savedLIBS"
                 AC_LINK_IFELSE( [AC_LANG_PROGRAM([@%:@include $OPENCL_HEADER],[clGetPlatformIDs(0,0,0);])],
                    [AC_MSG_RESULT($OPENCL_LIBS)],
                    [AC_MSG_RESULT([no])
                     if test "x$with_opencl" != xcheck; then
                         AC_MSG_FAILURE([OpenCL support was requested, but no usable OpenCL library was fund])
                     fi
                    ]
                 )
                ]
            )
            LIBS=$savedLIBS
            ]
        )

        AS_IF([test "x$OPENCL_HEADER" != "x" -a "x$OPENCL_LIBS" != "x"],
            [with_opencl=yes
             MPI_SUBDIRS="$MPI_SUBDIRS opencl"
             AC_SUBST([OPENCL_LIBS])
             ],
            [AC_MSG_NOTICE([no usable OpenCL installation found])
             with_opencl=no
            ])
    ])
AM_CONDITIONAL([BUILD_OPENCL], [test "x$with_opencl" = "xyes"])


# CUDA support
AC_ARG_WITH([cuda],
    [AS_HELP_STRING([--with-cuda],
        [build CUDA versions of SHOC benchmark programs @<:@default=check@:>@])],
    [],
    [with_cuda=check])

USE_CUDA=no
AS_IF([test "x$with_cuda" != xno],
    [
        # Check for the nvcc program
        AC_PATH_PROG([NVCC],[nvcc],[])

        AS_IF([test "x$NVCC" != x],
            [ savedCXX=$CXX
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

              AC_CHECK_HEADERS([cuda.h cuda_runtime.h],[],
                [NVCC=""]
              )
              AC_CHECK_LIB([cublas], [cublasInit],[],
                [NVCC=""]
              )
              AC_CHECK_LIB([cufft], [cufftPlan1d], [],
                [NVCC=""]
              )

              CPPFLAGS=$savedCPPFLAGS
              LIBS=$savedLIBS
              CXX=$savedCXX
              CXXFLAGS=$savedCXXFLAGS
              LDFLAGS=$savedLDFLAGS
            ],
            [ if test "x$with_cuda" != xcheck; then
                AC_MSG_FAILURE([CUDA support was requested, but no usable CUDA installation found])
              fi
              NVCC=""
            ]
        )

        AS_IF([test "x$NVCC" != x],
            [with_cuda=yes
             USE_CUDA=yes
             MPI_SUBDIRS="$MPI_SUBDIRS cuda"
             AC_SUBST([CUDA_INCDIR])],
            [AC_MSG_NOTICE([no usable CUDA installation found])
             with_cuda=no
            ])

        AS_IF([test "x$with_cuda" = xyes],
            [AS_IF([test "x$CUDA_CPPFLAGS" = x],
                [dnl generate CUDA code for broad spectrum of devices
                 CUDA_CPPFLAGS=["-gencode=arch=compute_60,code=sm_60"]
                 AC_MSG_NOTICE([using CUDA_CPPFLAGS=$CUDA_CPPFLAGS])
                 AC_MSG_WARN([*** SHOC programs may fail if these -gencode values do not apply to your GPU.  If they don't apply, or if you want to shorten SHOC's compile time, determine the -gencode options that apply for your GPU and re-configure SHOC explicitly specifying only those options in CUDA_CPPFLAGS during the configure. ***])
                ]
             )
             AC_SUBST([CUDA_CPPFLAGS])
            ]
        )
    ]
)
AM_CONDITIONAL([BUILD_CUDA], [test "x$with_cuda" = "xyes"])

AC_CONFIG_FILES([Makefile
                 config/common.mk
                 config/config.mk
                 config/dirtargets.mk
                 config/targets.mk
                 src/Makefile
                 src/common/Makefile
                 src/cuda/Makefile
                 src/cuda/level0/Makefile
                 ])
AC_OUTPUT

