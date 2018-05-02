/* file: mkl_vsl_defines.h */
/*
//                             INTEL CONFIDENTIAL
//  Copyright(C) 2006-2008 Intel Corporation. All Rights Reserved.
//  The source code contained  or  described herein and all documents related to
//  the source code ("Material") are owned by Intel Corporation or its suppliers
//  or licensors.  Title to the  Material remains with  Intel Corporation or its
//  suppliers and licensors. The Material contains trade secrets and proprietary
//  and  confidential  information of  Intel or its suppliers and licensors. The
//  Material  is  protected  by  worldwide  copyright  and trade secret laws and
//  treaty  provisions. No part of the Material may be used, copied, reproduced,
//  modified, published, uploaded, posted, transmitted, distributed or disclosed
//  in any way without Intel's prior express written permission.
//  No license  under any  patent, copyright, trade secret or other intellectual
//  property right is granted to or conferred upon you by disclosure or delivery
//  of the Materials,  either expressly, by implication, inducement, estoppel or
//  otherwise.  Any  license  under  such  intellectual property  rights must be
//  express and approved by Intel in writing.
*/
/*
//++
//  User-level macro definitions
//--
*/

#ifndef __MKL_VSL_DEFINES_H__
#define __MKL_VSL_DEFINES_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/*
// "No error" status
*/
#define VSL_STATUS_OK                      0
#define VSL_ERROR_OK                       0

/*
// Common errors (-1..-999)
*/
#define VSL_ERROR_FEATURE_NOT_IMPLEMENTED  -1
#define VSL_ERROR_UNKNOWN                  -2
#define VSL_ERROR_BADARGS                  -3
#define VSL_ERROR_MEM_FAILURE              -4
#define VSL_ERROR_NULL_PTR                 -5


/*
// RNG errors (-1000..-1999)
*/
/* brng errors */
#define VSL_ERROR_INVALID_BRNG_INDEX        -1000
#define VSL_ERROR_LEAPFROG_UNSUPPORTED      -1002
#define VSL_ERROR_SKIPAHEAD_UNSUPPORTED     -1003
#define VSL_ERROR_BRNGS_INCOMPATIBLE        -1005
#define VSL_ERROR_BAD_STREAM                -1006
#define VSL_ERROR_BRNG_TABLE_FULL           -1007
#define VSL_ERROR_BAD_STREAM_STATE_SIZE     -1008
#define VSL_ERROR_BAD_WORD_SIZE             -1009
#define VSL_ERROR_BAD_NSEEDS                -1010
#define VSL_ERROR_BAD_NBITS                 -1011
#define VSL_QRNG_PERIOD_ELAPSED             -1012
#define VSL_ERROR_LEAPFROG_NSTREAMS_TOO_BIG -1013

/* abstract stream related errors */
#define VSL_ERROR_BAD_UPDATE                -1120
#define VSL_ERROR_NO_NUMBERS                -1121
#define VSL_ERROR_INVALID_ABSTRACT_STREAM   -1122

/* read/write stream to file errors */
#define VSL_ERROR_FILE_CLOSE                -1100
#define VSL_ERROR_FILE_OPEN                 -1101
#define VSL_ERROR_FILE_WRITE                -1102
#define VSL_ERROR_FILE_READ                 -1103

#define VSL_ERROR_BAD_FILE_FORMAT           -1110
#define VSL_ERROR_UNSUPPORTED_FILE_VER      -1111

/* Convolution/correlation errors */
#define VSL_CC_ERROR_NOT_IMPLEMENTED        (-2000)
#define VSL_CC_ERROR_ALLOCATION_FAILURE     (-2001)
#define VSL_CC_ERROR_BAD_DESCRIPTOR         (-2200)
#define VSL_CC_ERROR_SERVICE_FAILURE        (-2210)
#define VSL_CC_ERROR_EDIT_FAILURE           (-2211)
#define VSL_CC_ERROR_EDIT_PROHIBITED        (-2212)
#define VSL_CC_ERROR_COMMIT_FAILURE         (-2220)
#define VSL_CC_ERROR_COPY_FAILURE           (-2230)
#define VSL_CC_ERROR_DELETE_FAILURE         (-2240)
#define VSL_CC_ERROR_BAD_ARGUMENT           (-2300)
#define VSL_CC_ERROR_DIMS                   (-2301)
#define VSL_CC_ERROR_START                  (-2302)
#define VSL_CC_ERROR_DECIMATION             (-2303)
#define VSL_CC_ERROR_XSHAPE                 (-2311)
#define VSL_CC_ERROR_YSHAPE                 (-2312)
#define VSL_CC_ERROR_ZSHAPE                 (-2313)
#define VSL_CC_ERROR_XSTRIDE                (-2321)
#define VSL_CC_ERROR_YSTRIDE                (-2322)
#define VSL_CC_ERROR_ZSTRIDE                (-2323)
#define VSL_CC_ERROR_X                      (-2331)
#define VSL_CC_ERROR_Y                      (-2332)
#define VSL_CC_ERROR_Z                      (-2333)
#define VSL_CC_ERROR_JOB                    (-2100)
#define VSL_CC_ERROR_KIND                   (-2110)
#define VSL_CC_ERROR_MODE                   (-2120)
#define VSL_CC_ERROR_TYPE                   (-2130)
#define VSL_CC_ERROR_PRECISION              (-2400)
#define VSL_CC_ERROR_EXTERNAL_PRECISION     (-2141)
#define VSL_CC_ERROR_INTERNAL_PRECISION     (-2142)
#define VSL_CC_ERROR_METHOD                 (-2400)
#define VSL_CC_ERROR_OTHER                  (-2800)

/*
// CONV/CORR RELATED MACRO DEFINITIONS
*/
#define VSL_CONV_MODE_AUTO        0
#define VSL_CORR_MODE_AUTO        0
#define VSL_CONV_MODE_DIRECT      1
#define VSL_CORR_MODE_DIRECT      1
#define VSL_CONV_MODE_FFT         2
#define VSL_CORR_MODE_FFT         2
#define VSL_CONV_PRECISION_SINGLE 1
#define VSL_CORR_PRECISION_SINGLE 1
#define VSL_CONV_PRECISION_DOUBLE 2
#define VSL_CORR_PRECISION_DOUBLE 2

/*
//++
//  BASIC RANDOM NUMBER GENERATOR (BRNG) RELATED MACRO DEFINITIONS
//--
*/

/*
//  MAX NUMBER OF BRNGS CAN BE REGISTERED IN VSL
//  No more than VSL_MAX_REG_BRNGS basic generators can be registered in VSL
//  (including predefined basic generators).
//
//  Change this number to increase/decrease number of BRNGs can be registered.
*/
#define VSL_MAX_REG_BRNGS           512

/*
//  PREDEFINED BRNG NAMES
*/
#define VSL_BRNG_SHIFT      20
#define VSL_BRNG_INC        (1<<VSL_BRNG_SHIFT)

#define VSL_BRNG_MCG31      (VSL_BRNG_INC)
#define VSL_BRNG_R250       (VSL_BRNG_MCG31    +VSL_BRNG_INC)
#define VSL_BRNG_MRG32K3A   (VSL_BRNG_R250     +VSL_BRNG_INC)
#define VSL_BRNG_MCG59      (VSL_BRNG_MRG32K3A +VSL_BRNG_INC)
#define VSL_BRNG_WH         (VSL_BRNG_MCG59    +VSL_BRNG_INC)
#define VSL_BRNG_SOBOL      (VSL_BRNG_WH       +VSL_BRNG_INC)
#define VSL_BRNG_NIEDERR    (VSL_BRNG_SOBOL    +VSL_BRNG_INC)
#define VSL_BRNG_MT19937    (VSL_BRNG_NIEDERR  +VSL_BRNG_INC)
#define VSL_BRNG_MT2203     (VSL_BRNG_MT19937  +VSL_BRNG_INC)
#define VSL_BRNG_IABSTRACT  (VSL_BRNG_MT2203   +VSL_BRNG_INC)
#define VSL_BRNG_DABSTRACT  (VSL_BRNG_IABSTRACT+VSL_BRNG_INC)
#define VSL_BRNG_SABSTRACT  (VSL_BRNG_DABSTRACT+VSL_BRNG_INC)


/*
//  LEAPFROG METHOD FOR GRAY-CODE BASED QUASI-RANDOM NUMBER BASIC GENERATORS
//  VSL_BRNG_SOBOL and VSL_BRNG_NIEDERR are Gray-code based quasi-random number
//  basic generators. In contrast to pseudorandom number basic generators,
//  quasi-random ones take the dimension as initialization parameter.
//
//  Suppose that quasi-random number generator (QRNG) dimension is S. QRNG
//  sequence is a sequence of S-dimensional vectors:
//
//     x0=(x0[0],x0[1],...,x0[S-1]),x1=(x1[0],x1[1],...,x1[S-1]),...
//
//  VSL treats the output of any basic generator as 1-dimensional, however:
//
//     x0[0],x0[1],...,x0[S-1],x1[0],x1[1],...,x1[S-1],...
//
//  Because of nature of VSL_BRNG_SOBOL and VSL_BRNG_NIEDERR QRNGs,
//  the only S-stride Leapfrog method is supported for them. In other words,
//  user can generate subsequences, which consist of fixed elements of
//  vectors x0,x1,... For example, if 0 element is fixed, the following
//  subsequence is generated:
//
//     x0[1],x1[1],x2[1],...
//
//  To use the s-stride Leapfrog method with given QRNG, user should call
//  vslLeapfrogStream function with parameter k equal to element to be fixed
//  (0<=k<S) and parameter nstreams equal to VSL_QRNG_LEAPFROG_COMPONENTS.
*/
#define VSL_QRNG_LEAPFROG_COMPONENTS    0x7fffffff

/*
//  USER-DEFINED PARAMETERS FOR QUASI-RANDOM NUMBER BASIC GENERATORS
//  VSL_BRNG_SOBOL and VSL_BRNG_NIEDERR are Gray-code based quasi-random
//  number basic generators. Default parameters of the generators
//  support generation of quasi-random number vectors of dimensions
//  S<=40 for SOBOL and S<=318 for NIEDERRITER. The library provides
//  opportunity to register user-defined initial values for the
//  generators and generate quasi-random vectors of desirable dimension.
//  There is also opportunity to register user-defined parameters for
//  default dimensions and obtain another sequence of quasi-random vectors.
//  Service function vslNewStreamEx is used to pass the parameters to
//  the library. Data are packed into array params, parameter of the routine.
//  First element of the array is used for dimension S, second element
//  contains indicator, VSL_USER_QRNG_INITIAL_VALUES, of user-defined
//  parameters for quasi-random number generators.
//  Macros VSL_USER_PRIMITIVE_POLYMS and VSL_USER_INIT_DIRECTION_NUMBERS
//  are used to describe which data are passed to SOBOL QRNG and
//  VSL_USER_IRRED_POLYMS - which data are passed to NIEDERRITER QRNG.
//  For example, to demonstrate that both primitive polynomials and initial
//  direction numbers are passed in SOBOL one should set third element of the
//  array params to  VSL_USER_PRIMITIVE_POLYMS | VSL_USER_DIRECTION_NUMBERS.
//  Macro VSL_QRNG_OVERRIDE_1ST_DIM_INIT is used to override default
//  initialization for the first dimension. Macro VSL_USER_DIRECTION_NUMBERS
//  is used when direction numbers calculated on the user side are passed
//  into the generators. More detailed description of interface for
//  registration of user-defined QRNG initial parameters can be found
//  in VslNotes.pdf.
*/
#define VSL_USER_QRNG_INITIAL_VALUES     0x1
#define VSL_USER_PRIMITIVE_POLYMS        0x1
#define VSL_USER_INIT_DIRECTION_NUMBERS  0x2
#define VSL_USER_IRRED_POLYMS            0x1
#define VSL_USER_DIRECTION_NUMBERS       0x4
#define VSL_QRNG_OVERRIDE_1ST_DIM_INIT   0x8


/*
//  INITIALIZATION METHODS FOR USER-DESIGNED BASIC RANDOM NUMBER GENERATORS.
//  Each BRNG must support at least VSL_INIT_METHOD_STANDARD initialization
//  method. In addition, VSL_INIT_METHOD_LEAPFROG and VSL_INIT_METHOD_SKIPAHEAD
//  initialization methods can be supported.
//
//  If VSL_INIT_METHOD_LEAPFROG is not supported then initialization routine
//  must return VSL_ERROR_LEAPFROG_UNSUPPORTED error code.
//
//  If VSL_INIT_METHOD_SKIPAHEAD is not supported then initialization routine
//  must return VSL_ERROR_SKIPAHEAD_UNSUPPORTED error code.
//
//  If there is no error during initialization, the initialization routine must
//  return VSL_ERROR_OK code.
*/
#define VSL_INIT_METHOD_STANDARD  0
#define VSL_INIT_METHOD_LEAPFROG  1
#define VSL_INIT_METHOD_SKIPAHEAD 2

/*
//++
//  ACCURACY FLAG FOR DISTRIBUTION GENERATORS
//  This flag defines mode of random number generation.
//  If accuracy mode is set distribution generators will produce
//  numbers lying exactly within definitional domain for all values
//  of distribution parameters. In this case slight performance
//  degradation is expected. By default accuracy mode is switched off
//  admitting random numbers to be out of the definitional domain for
//  specific values of distribution parameters.
//  This macro is used to form names for accuracy versions of
//  distribution number generators
//--
*/
#define VSL_METHOD_ACCURACY_FLAG (1<<30)

/*
//++
//  TRANSFORMATION METHOD NAMES FOR DISTRIBUTION RANDOM NUMBER GENERATORS
//  VSL interface allows more than one generation method in a distribution
//  transformation subroutine. Following macro definitions are used to
//  specify generation method for given distribution generator.
//
//  Method name macro is constructed as
//
//     VSL_METHOD_<Precision><Distribution>_<Method>
//
//  where
//
//     <Precision> - S (single precision) or D (double precision)
//     <Distribution> - probability distribution
//     <Method> - method name
//
//  VSL_METHOD_<Precision><Distribution>_<Method> should be used with
//  vsl<precision>Rng<Distribution> function only, where
//
//     <precision> - s (single) or d (double)
//     <Distribution> - probability distribution
//--
*/

/*
// Uniform
//
// <Method>   <Short Description>
// STD        standard method. Currently there is only one method for this
//            distribution generator
*/
#define VSL_METHOD_SUNIFORM_STD 0 /* vslsRngUniform */
#define VSL_METHOD_DUNIFORM_STD 0 /* vsldRngUniform */
#define VSL_METHOD_IUNIFORM_STD 0 /* vsliRngUniform */

#define VSL_METHOD_SUNIFORM_STD_ACCURATE \
  VSL_METHOD_SUNIFORM_STD | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vslsRngUniform */

#define VSL_METHOD_DUNIFORM_STD_ACCURATE \
  VSL_METHOD_DUNIFORM_STD | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vsldRngUniform */

/*
// Uniform Bits
//
// <Method>   <Short Description>
// STD        standard method. Currently there is only one method for this
//            distribution generator
*/
#define VSL_METHOD_IUNIFORMBITS_STD 0 /* vsliRngUniformBits */

/*
// Gaussian
//
// <Method>   <Short Description>
// BOXMULLER  generates normally distributed random number x thru the pair of
//            uniformly distributed numbers u1 and u2 according to the formula:
//
//               x=sqrt(-ln(u1))*sin(2*Pi*u2)
//
// BOXMULLER2 generates pair of normally distributed random numbers x1 and x2
//            thru the pair of uniformly dustributed numbers u1 and u2
//            according to the formula
//
//               x1=sqrt(-ln(u1))*sin(2*Pi*u2)
//               x2=sqrt(-ln(u1))*cos(2*Pi*u2)
//
//            NOTE: implementation correctly works with odd vector lengths
//
// ICDF       inverse cumulative distribution function method
*/
#define VSL_METHOD_SGAUSSIAN_BOXMULLER   0 /* vslsRngGaussian */
#define VSL_METHOD_SGAUSSIAN_BOXMULLER2  1 /* vslsRngGaussian */
#define VSL_METHOD_SGAUSSIAN_ICDF        2 /* vslsRngGaussian */
#define VSL_METHOD_DGAUSSIAN_BOXMULLER   0 /* vsldRngGaussian */
#define VSL_METHOD_DGAUSSIAN_BOXMULLER2  1 /* vsldRngGaussian */
#define VSL_METHOD_DGAUSSIAN_ICDF        2 /* vsldRngGaussian */

/*
// GaussianMV - multivariate (correlated) normal
// Multivariate (correlated) normal random number generator is based on
// uncorrelated Gaussian random number generator (see vslsRngGaussian and
// vsldRngGaussian functions):
//
// <Method>   <Short Description>
// BOXMULLER  generates normally distributed random number x thru the pair of
//            uniformly distributed numbers u1 and u2 according to the formula:
//
//               x=sqrt(-ln(u1))*sin(2*Pi*u2)
//
// BOXMULLER2 generates pair of normally distributed random numbers x1 and x2
//            thru the pair of uniformly dustributed numbers u1 and u2
//            according to the formula
//
//               x1=sqrt(-ln(u1))*sin(2*Pi*u2)
//               x2=sqrt(-ln(u1))*cos(2*Pi*u2)
//
//            NOTE: implementation correctly works with odd vector lengths
//
// ICDF       inverse cumulative distribution function method
*/
#define VSL_METHOD_SGAUSSIANMV_BOXMULLER   0 /* vslsRngGaussianMV */
#define VSL_METHOD_SGAUSSIANMV_BOXMULLER2  1 /* vslsRngGaussianMV */
#define VSL_METHOD_SGAUSSIANMV_ICDF        2 /* vslsRngGaussianMV */
#define VSL_METHOD_DGAUSSIANMV_BOXMULLER   0 /* vsldRngGaussianMV */
#define VSL_METHOD_DGAUSSIANMV_BOXMULLER2  1 /* vsldRngGaussianMV */
#define VSL_METHOD_DGAUSSIANMV_ICDF        2 /* vsldRngGaussianMV */

/*
// Exponential
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_METHOD_SEXPONENTIAL_ICDF 0 /* vslsRngExponential */
#define VSL_METHOD_DEXPONENTIAL_ICDF 0 /* vsldRngExponential */

#define VSL_METHOD_SEXPONENTIAL_ICDF_ACCURATE \
   VSL_METHOD_SEXPONENTIAL_ICDF | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vslsRngExponential */

#define VSL_METHOD_DEXPONENTIAL_ICDF_ACCURATE \
   VSL_METHOD_DEXPONENTIAL_ICDF | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vsldRngExponential */

/*
// Laplace
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
//
// ICDF - inverse cumulative distribution function method:
//
//           x=+/-ln(u) with probability 1/2,
//
//        where
//
//           x - random number with Laplace distribution,
//           u - uniformly distributed random number
*/
#define VSL_METHOD_SLAPLACE_ICDF 0 /* vsldRngLaplace */
#define VSL_METHOD_DLAPLACE_ICDF 0 /* vsldRngLaplace */

/*
// Weibull
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_METHOD_SWEIBULL_ICDF 0 /* vslsRngWeibull */
#define VSL_METHOD_DWEIBULL_ICDF 0 /* vsldRngWeibull */

#define VSL_METHOD_SWEIBULL_ICDF_ACCURATE \
   VSL_METHOD_SWEIBULL_ICDF | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vslsRngWeibull */

#define VSL_METHOD_DWEIBULL_ICDF_ACCURATE \
   VSL_METHOD_DWEIBULL_ICDF | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vsldRngWeibull */

/*
// Cauchy
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_METHOD_SCAUCHY_ICDF 0 /* vslsRngCauchy */
#define VSL_METHOD_DCAUCHY_ICDF 0 /* vsldRngCauchy */

/*
// Rayleigh
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_METHOD_SRAYLEIGH_ICDF 0 /* vslsRngRayleigh */
#define VSL_METHOD_DRAYLEIGH_ICDF 0 /* vsldRngRayleigh */

#define VSL_METHOD_SRAYLEIGH_ICDF_ACCURATE \
   VSL_METHOD_SRAYLEIGH_ICDF | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vslsRngRayleigh */

#define VSL_METHOD_DRAYLEIGH_ICDF_ACCURATE \
   VSL_METHOD_DRAYLEIGH_ICDF | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vsldRngRayleigh */


/*
// Lognormal
//
// <Method>   <Short Description>
// BOXMULLER2       Box-Muller 2 algorithm based method
*/
#define VSL_METHOD_SLOGNORMAL_BOXMULLER2 0 /* vslsRngLognormal */
#define VSL_METHOD_DLOGNORMAL_BOXMULLER2 0 /* vsldRngLognormal */
#define VSL_METHOD_SLOGNORMAL_ICDF 0       /* vslsRngLognormal */
#define VSL_METHOD_DLOGNORMAL_ICDF 0       /* vsldRngLognormal */

#define VSL_METHOD_SLOGNORMAL_BOXMULLER2_ACCURATE \
   VSL_METHOD_SLOGNORMAL_BOXMULLER2 | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vslsRngLognormal */

#define VSL_METHOD_DLOGNORMAL_BOXMULLER2_ACCURATE \
   VSL_METHOD_DLOGNORMAL_BOXMULLER2 | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vsldRngLognormal */

#define VSL_METHOD_SLOGNORMAL_ICDF_ACCURATE \
   VSL_METHOD_SLOGNORMAL_ICDF | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vslsRngLognormal */

#define VSL_METHOD_DLOGNORMAL_ICDF_ACCURATE \
   VSL_METHOD_DLOGNORMAL_ICDF | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vsldRngLognormal */

/*
// Gumbel
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_METHOD_SGUMBEL_ICDF 0 /* vslsRngGumbel */
#define VSL_METHOD_DGUMBEL_ICDF 0 /* vsldRngGumbel */

/*
// Gamma
//
// Comments:
// alpha>1             - algorithm of Marsaglia is used, nonlinear
//                       transformation of gaussian numbers based on
//                       acceptance/rejection method with squeezes;
// alpha>=0.6, alpha<1 - rejection from the Weibull distribution is used;
// alpha<0.6           - transformation of exponential power distribution
//                       (EPD) is used, EPD random numbers are generated
//                       by means of acceptance/rejection technique;
// alpha=1             - gamma distribution reduces to exponential
//                       distribution
*/
#define VSL_METHOD_SGAMMA_GNORM 0 /* vslsRngGamma */
#define VSL_METHOD_DGAMMA_GNORM 0 /* vsldRngGamma */

#define VSL_METHOD_SGAMMA_GNORM_ACCURATE \
   VSL_METHOD_SGAMMA_GNORM | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vslsRngGamma */

#define VSL_METHOD_DGAMMA_GNORM_ACCURATE \
   VSL_METHOD_DGAMMA_GNORM | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vsldRngGamma */


/*
// Beta
//
// Comments:
// CJA - stands for first letters of Cheng, Johnk, and Atkinson.
// Cheng    - for min(p,q) > 1 method of Cheng,
//            generation of beta random numbers of the second kind
//            based on acceptance/rejection technique and its
//            transformation to beta random numbers of the first kind;
// Johnk    - for max(p,q) < 1 methods of Johnk and Atkinson:
//            if q + K*p^2+C<=0, K=0.852..., C=-0.956...
//            algorithm of Johnk:
//            beta distributed random number is generated as
//            u1^(1/p) / (u1^(1/p)+u2^(1/q)), if u1^(1/p)+u2^(1/q)<=1;
//            otherwise switching algorithm of Atkinson: interval (0,1)
//            is divided into two domains (0,t) and (t,1), on each interval
//            acceptance/rejection technique with convenient majorizing
//            function is used;
// Atkinson - for min(p,q)<1, max(p,q)>1 switching algorithm of Atkinson
//            is used (with another point t, see short description above);
// ICDF     - inverse cumulative distribution function method according
//            to formulas x=1-u^(1/q) for p = 1, and x = u^(1/p) for q=1,
//            where x is beta distributed random number,
//            u - uniformly distributed random number.
//            for p=q=1 beta distribution reduces to uniform distribution.
//
*/
#define VSL_METHOD_SBETA_CJA 0 /* vslsRngBeta */
#define VSL_METHOD_DBETA_CJA 0 /* vsldRngBeta */

#define VSL_METHOD_SBETA_CJA_ACCURATE \
   VSL_METHOD_SBETA_CJA | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vslsRngBeta */

#define VSL_METHOD_DBETA_CJA_ACCURATE \
   VSL_METHOD_DBETA_CJA | VSL_METHOD_ACCURACY_FLAG
    /* accurate mode of vsldRngBeta */

/*
// Bernoulli
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_METHOD_IBERNOULLI_ICDF 0 /* vsliRngBernoulli */

/*
// Geometric
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_METHOD_IGEOMETRIC_ICDF 0 /* vsliRngGeometric */

/*
// Binomial
//
// <Method>   <Short Description>
// BTPE       for ntrial*min(p,1-p)>30 acceptance/rejection method with
//            decomposition onto 4 regions:
//
//               * 2 parallelograms;
//               * triangle;
//               * left exponential tail;
//               * right exponential tail.
//
//            othewise table lookup method is used
*/
#define VSL_METHOD_IBINOMIAL_BTPE 0 /* vsliRngBinomial */

/*
// Hypergeometric
//
// <Method>   <Short Description>
// H2PE       if mode of distribution is large, acceptance/rejection method is
//            used with decomposition onto 3 regions:
//
//               * rectangular;
//               * left exponential tail;
//               * right exponential tail.
//
//            othewise table lookup method is used
*/
#define VSL_METHOD_IHYPERGEOMETRIC_H2PE 0 /* vsliRngHypergeometric */

/*
// Poisson
//
// <Method>   <Short Description>
// PTPE       if lambda>=27, acceptance/rejection method is used with
//            decomposition onto 4 regions:
//
//               * 2 parallelograms;
//               * triangle;
//               * left exponential tail;
//               * right exponential tail.
//
//            othewise table lookup method is used
//
// POISNORM   for lambda>=1 method is based on Poisson inverse CDF
//            approximation by Gaussian inverse CDF; for lambda<1
//            table lookup method is used.
*/
#define VSL_METHOD_IPOISSON_PTPE     0 /* vsliRngPoisson */
#define VSL_METHOD_IPOISSON_POISNORM 1 /* vsliRngPoisson */

/*
// Poisson
//
// <Method>   <Short Description>
// POISNORM   for lambda>=1 method is based on Poisson inverse CDF
//            approximation by Gaussian inverse CDF; for lambda<1
//            ICDF method is used.
*/
#define VSL_METHOD_IPOISSONV_POISNORM 0 /* vsliRngPoissonV */

/*
// Negbinomial
//
// <Method>   <Short Description>
// NBAR       if (a-1)*(1-p)/p>=100, acceptance/rejection method is used with
//            decomposition onto 5 regions:
//
//               * rectangular;
//               * 2 trapezoid;
//               * left exponential tail;
//               * right exponential tail.
//
//            othewise table lookup method is used.
*/
#define VSL_METHOD_INEGBINOMIAL_NBAR 0 /* vsliRngNegbinomial */


/*
//++
//  MATRIX STORAGE SCHEMES
//--
*/

/*
// Some multivariate random number generators, e.g. GaussianMV, operate
// with matrix parameters. To optimize matrix parameters usage VSL offers
// following matrix storage schemes. (See VSL documentation for more details).
//
// FULL     - whole matrix is stored
// PACKED   - lower/higher triangular matrix is packed in 1-dimensional array
// DIAGONAL - diagonal elements are packed in 1-dimensional array
*/
#define VSL_MATRIX_STORAGE_FULL     0
#define VSL_MATRIX_STORAGE_PACKED   1
#define VSL_MATRIX_STORAGE_DIAGONAL 2

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_VSL_DEFINES_H__ */
