/* file: mkl_vsl_functions.h */
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
//  User-level VSL function declarations
//--
*/

#ifndef __MKL_VSL_FUNCTIONS_H__
#define __MKL_VSL_FUNCTIONS_H__

#include "mkl_vsl_types.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/*
//++
//  EXTERNAL API MACROS.
//  Used to construct VSL function declaration. Change them if you are going to
//  provide different API for VSL functions.
//--
*/
#define _Vsl_Api(rtype,name,arg)   extern rtype name    arg;
#define _vsl_api(rtype,name,arg)   extern rtype name##_ arg;
#define _VSL_API(rtype,name,arg)   extern rtype name##_ arg;

/*
//++
//  VSL CONTINUOUS DISTRIBUTION GENERATOR FUNCTION DECLARATIONS.
//--
*/
/* Cauchy distribution */
_Vsl_Api(int,vdRngCauchy,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , double [], double  , double  ))
_VSL_API(int,VDRNGCAUCHY,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *))
_vsl_api(int,vdrngcauchy,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *))
_Vsl_Api(int,vsRngCauchy,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , float [],  float  ,  float   ))
_VSL_API(int,VSRNGCAUCHY,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float * ))
_vsl_api(int,vsrngcauchy,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float * ))

/* Uniform distribution */
_Vsl_Api(int,vdRngUniform,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , double [], double  , double  ))
_VSL_API(int,VDRNGUNIFORM,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *))
_vsl_api(int,vdrnguniform,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *))
_Vsl_Api(int,vsRngUniform,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , float [],  float  ,  float   ))
_VSL_API(int,VSRNGUNIFORM,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float * ))
_vsl_api(int,vsrnguniform,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float * ))

/* Gaussian distribution */
_Vsl_Api(int,vdRngGaussian,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , double [], double  , double  ))
_VSL_API(int,VDRNGGAUSSIAN,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *))
_vsl_api(int,vdrnggaussian,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *))
_Vsl_Api(int,vsRngGaussian,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , float [],  float  ,  float   ))
_VSL_API(int,VSRNGGAUSSIAN,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float * ))
_vsl_api(int,vsrnggaussian,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float * ))

/* GaussianMV distribution */
_Vsl_Api(int,vdRngGaussianMV,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , double [], MKL_INT  ,  MKL_INT  , double *, double *))
_VSL_API(int,VDRNGGAUSSIANMV,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], MKL_INT *,  MKL_INT *, double *, double *))
_vsl_api(int,vdrnggaussianmv,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], MKL_INT *,  MKL_INT *, double *, double *))
_Vsl_Api(int,vsRngGaussianMV,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , float [],  MKL_INT  ,  MKL_INT  , float *,  float * ))
_VSL_API(int,VSRNGGAUSSIANMV,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  MKL_INT *,  MKL_INT *, float *,  float * ))
_vsl_api(int,vsrnggaussianmv,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  MKL_INT *,  MKL_INT *, float *,  float * ))

/* Exponential distribution */
_Vsl_Api(int,vdRngExponential,(MKL_INT  , VSLStreamStatePtr , MKL_INT  ,  double [], double  , double  ))
_VSL_API(int,VDRNGEXPONENTIAL,(MKL_INT *, VSLStreamStatePtr , MKL_INT *,  double [], double *, double *))
_vsl_api(int,vdrngexponential,(MKL_INT *, VSLStreamStatePtr , MKL_INT *,  double [], double *, double *))
_Vsl_Api(int,vsRngExponential,(MKL_INT  , VSLStreamStatePtr , MKL_INT  ,  float [],  float  ,  float   ))
_VSL_API(int,VSRNGEXPONENTIAL,(MKL_INT *, VSLStreamStatePtr , MKL_INT *,  float [],  float *,  float * ))
_vsl_api(int,vsrngexponential,(MKL_INT *, VSLStreamStatePtr , MKL_INT *,  float [],  float *,  float * ))

/* Laplace distribution */
_Vsl_Api(int,vdRngLaplace,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , double [], double  , double  ))
_VSL_API(int,VDRNGLAPLACE,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *))
_vsl_api(int,vdrnglaplace,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *))
_Vsl_Api(int,vsRngLaplace,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , float [],  float  ,  float   ))
_VSL_API(int,VSRNGLAPLACE,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float * ))
_vsl_api(int,vsrnglaplace,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float * ))

/* Weibull distribution */
_Vsl_Api(int,vdRngWeibull,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , double [], double  , double  , double  ))
_VSL_API(int,VDRNGWEIBULL,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *, double *))
_vsl_api(int,vdrngweibull,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *, double *))
_Vsl_Api(int,vsRngWeibull,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , float [],  float  ,  float  ,  float   ))
_VSL_API(int,VSRNGWEIBULL,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float *,  float * ))
_vsl_api(int,vsrngweibull,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float *,  float * ))

/* Rayleigh distribution */
_Vsl_Api(int,vdRngRayleigh,(MKL_INT  , VSLStreamStatePtr , MKL_INT  ,  double [], double  , double  ))
_VSL_API(int,VDRNGRAYLEIGH,(MKL_INT *, VSLStreamStatePtr , MKL_INT *,  double [], double *, double *))
_vsl_api(int,vdrngrayleigh,(MKL_INT *, VSLStreamStatePtr , MKL_INT *,  double [], double *, double *))
_Vsl_Api(int,vsRngRayleigh,(MKL_INT  , VSLStreamStatePtr , MKL_INT  ,  float [],  float  ,  float   ))
_VSL_API(int,VSRNGRAYLEIGH,(MKL_INT *, VSLStreamStatePtr , MKL_INT *,  float [],  float *,  float * ))
_vsl_api(int,vsrngrayleigh,(MKL_INT *, VSLStreamStatePtr , MKL_INT *,  float [],  float *,  float * ))

/* Lognormal distribution */
_Vsl_Api(int,vdRngLognormal,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , double [], double  , double  , double  , double  ))
_VSL_API(int,VDRNGLOGNORMAL,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *, double *, double *))
_vsl_api(int,vdrnglognormal,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *, double *, double *))
_Vsl_Api(int,vsRngLognormal,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , float [],  float  ,  float  ,  float  ,  float   ))
_VSL_API(int,VSRNGLOGNORMAL,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float *,  float *,  float * ))
_vsl_api(int,vsrnglognormal,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float *,  float *,  float * ))

/* Gumbel distribution */
_Vsl_Api(int,vdRngGumbel,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , double [], double  , double  ))
_VSL_API(int,VDRNGGUMBEL,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *))
_vsl_api(int,vdrnggumbel,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *))
_Vsl_Api(int,vsRngGumbel,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , float [],  float  ,  float   ))
_VSL_API(int,VSRNGGUMBEL,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float * ))
_vsl_api(int,vsrnggumbel,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float * ))

/* Gamma distribution */
_Vsl_Api(int,vdRngGamma,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , double [], double  , double  , double  ))
_VSL_API(int,VDRNGGAMMA,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *, double *))
_vsl_api(int,vdrnggamma,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *, double *))
_Vsl_Api(int,vsRngGamma,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , float [],  float  ,  float  ,  float   ))
_VSL_API(int,VSRNGGAMMA,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float *,  float * ))
_vsl_api(int,vsrnggamma,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float *,  float * ))

/* Beta distribution */
_Vsl_Api(int,vdRngBeta,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , double [], double  , double  , double  , double  ))
_VSL_API(int,VDRNGBETA,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *, double *, double *))
_vsl_api(int,vdrngbeta,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, double [], double *, double *, double *, double *))
_Vsl_Api(int,vsRngBeta,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , float [],  float  ,  float  ,  float  ,  float   ))
_VSL_API(int,VSRNGBETA,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float *,  float *,  float * ))
_vsl_api(int,vsrngbeta,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, float [],  float *,  float *,  float *,  float * ))

/*
//++
//  VSL DISCRETE DISTRIBUTION GENERATOR FUNCTION DECLARATIONS.
//--
*/
/* Bernoulli distribution */
_Vsl_Api(int,viRngBernoulli,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , int [], double  ))
_VSL_API(int,VIRNGBERNOULLI,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], double *))
_vsl_api(int,virngbernoulli,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], double *))

/* Uniform distribution */
_Vsl_Api(int,viRngUniform,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , int [], int  , int  ))
_VSL_API(int,VIRNGUNIFORM,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], int *, int *))
_vsl_api(int,virnguniform,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], int *, int *))

/* UniformBits distribution */
_Vsl_Api(int,viRngUniformBits,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , unsigned int []))
_VSL_API(int,VIRNGUNIFORMBITS,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, unsigned int []))
_vsl_api(int,virnguniformbits,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, unsigned int []))

/* Geometric distribution */
_Vsl_Api(int,viRngGeometric,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , int [], double  ))
_VSL_API(int,VIRNGGEOMETRIC,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], double *))
_vsl_api(int,virnggeometric,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], double *))

/* Binomial distribution */
_Vsl_Api(int,viRngBinomial,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , int [], int  , double  ))
_VSL_API(int,VIRNGBINOMIAL,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], int *, double *))
_vsl_api(int,virngbinomial,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], int *, double *))

/* Hypergeometric distribution */
_Vsl_Api(int,viRngHypergeometric,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , int [], int  , int  , int  ))
_VSL_API(int,VIRNGHYPERGEOMETRIC,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], int *, int *, int *))
_vsl_api(int,virnghypergeometric,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], int *, int *, int *))

/* Negbinomial distribution */
_Vsl_Api(int,viRngNegbinomial,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , int [], double  , double  ))
_Vsl_Api(int,viRngNegBinomial,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , int [], double  , double  ))
_VSL_API(int,VIRNGNEGBINOMIAL,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], double *, double *))
_vsl_api(int,virngnegbinomial,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], double *, double *))

/* Poisson distribution */
_Vsl_Api(int,viRngPoisson,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , int [], double  ))
_VSL_API(int,VIRNGPOISSON,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], double *))
_vsl_api(int,virngpoisson,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], double *))

/* PoissonV distribution */
_Vsl_Api(int,viRngPoissonV,(MKL_INT  , VSLStreamStatePtr , MKL_INT  , int [], double []))
_VSL_API(int,VIRNGPOISSONV,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], double []))
_vsl_api(int,virngpoissonv,(MKL_INT *, VSLStreamStatePtr , MKL_INT *, int [], double []))


/*
//++
//  VSL SERVICE FUNCTION DECLARATIONS.
//--
*/
/* NewStream - stream creation/initialization */
_Vsl_Api(int,vslNewStream,(VSLStreamStatePtr* , MKL_INT  , unsigned MKL_INT  ))
_vsl_api(int,vslnewstream,(VSLStreamStatePtr* , MKL_INT *, unsigned MKL_INT *))
_VSL_API(int,VSLNEWSTREAM,(VSLStreamStatePtr* , MKL_INT *, unsigned MKL_INT *))

/* NewStreamEx - advanced stream creation/initialization */
_Vsl_Api(int,vslNewStreamEx,(VSLStreamStatePtr* , MKL_INT  , MKL_INT  , const unsigned int[]))
_vsl_api(int,vslnewstreamex,(VSLStreamStatePtr* , MKL_INT *, MKL_INT *, const unsigned int[]))
_VSL_API(int,VSLNEWSTREAMEX,(VSLStreamStatePtr* , MKL_INT *, MKL_INT *, const unsigned int[]))

_Vsl_Api(int,vsliNewAbstractStream,(VSLStreamStatePtr* , MKL_INT  , unsigned int[], iUpdateFuncPtr))
_vsl_api(int,vslinewabstractstream,(VSLStreamStatePtr* , MKL_INT *, unsigned int[], iUpdateFuncPtr))
_VSL_API(int,VSLINEWABSTRACTSTREAM,(VSLStreamStatePtr* , MKL_INT *, unsigned int[], iUpdateFuncPtr))

_Vsl_Api(int,vsldNewAbstractStream,(VSLStreamStatePtr* , MKL_INT  , double[], double  , double  , dUpdateFuncPtr))
_vsl_api(int,vsldnewabstractstream,(VSLStreamStatePtr* , MKL_INT *, double[], double *, double *, dUpdateFuncPtr))
_VSL_API(int,VSLDNEWABSTRACTSTREAM,(VSLStreamStatePtr* , MKL_INT *, double[], double *, double *, dUpdateFuncPtr))

_Vsl_Api(int,vslsNewAbstractStream,(VSLStreamStatePtr* , MKL_INT  ,  float[], float  , float  , sUpdateFuncPtr))
_vsl_api(int,vslsnewabstractstream,(VSLStreamStatePtr* , MKL_INT *,  float[], float *, float *, sUpdateFuncPtr))
_VSL_API(int,VSLSNEWABSTRACTSTREAM,(VSLStreamStatePtr* , MKL_INT *,  float[], float *, float *, sUpdateFuncPtr))

/* DeleteStream - delete stream */
_Vsl_Api(int,vslDeleteStream,(VSLStreamStatePtr*))
_vsl_api(int,vsldeletestream,(VSLStreamStatePtr*))
_VSL_API(int,VSLDELETESTREAM,(VSLStreamStatePtr*))

/* CopyStream - copy all stream information */
_Vsl_Api(int,vslCopyStream,(VSLStreamStatePtr*, VSLStreamStatePtr))
_vsl_api(int,vslcopystream,(VSLStreamStatePtr*, VSLStreamStatePtr))
_VSL_API(int,VSLCOPYSTREAM,(VSLStreamStatePtr*, VSLStreamStatePtr))

/* CopyStreamState - copy stream state only */
_Vsl_Api(int,vslCopyStreamState,(VSLStreamStatePtr, VSLStreamStatePtr))
_vsl_api(int,vslcopystreamstate,(VSLStreamStatePtr, VSLStreamStatePtr))
_VSL_API(int,VSLCOPYSTREAMSTATE,(VSLStreamStatePtr, VSLStreamStatePtr))

/* LeapfrogStream - leapfrog method */
_Vsl_Api(int,vslLeapfrogStream,(VSLStreamStatePtr, MKL_INT  , MKL_INT  ))
_vsl_api(int,vslleapfrogstream,(VSLStreamStatePtr, MKL_INT *, MKL_INT *))
_VSL_API(int,VSLLEAPFROGSTREAM,(VSLStreamStatePtr, MKL_INT *, MKL_INT *))

/* SkipAheadStream - skip-ahead method */
_Vsl_Api(int,vslSkipAheadStream,(VSLStreamStatePtr, long long int  ))
_vsl_api(int,vslskipaheadstream,(VSLStreamStatePtr, long long int *))
_VSL_API(int,VSLSKIPAHEADSTREAM,(VSLStreamStatePtr, long long int *))

/* GetStreamStateBrng - get BRNG associated with given stream */
_Vsl_Api(int,vslGetStreamStateBrng,(VSLStreamStatePtr  ))
_vsl_api(int,vslgetstreamstatebrng,(VSLStreamStatePtr *))
_VSL_API(int,VSLGETSTREAMSTATEBRNG,(VSLStreamStatePtr *))

/* GetNumRegBrngs - get number of registered BRNGs */
_Vsl_Api(int,vslGetNumRegBrngs,(void))
_vsl_api(int,vslgetnumregbrngs,(void))
_VSL_API(int,VSLGETNUMREGBRNGS,(void))

/* RegisterBrng - register new BRNG */
_Vsl_Api(int,vslRegisterBrng,(const VSLBRngProperties* ))
_vsl_api(int,vslregisterbrng,(const VSLBRngProperties* ))
_VSL_API(int,VSLREGISTERBRNG,(const VSLBRngProperties* ))

/* GetBrngProperties - get BRNG properties */
_Vsl_Api(int,vslGetBrngProperties,(int  , VSLBRngProperties* ))
_vsl_api(int,vslgetbrngproperties,(int *, VSLBRngProperties* ))
_VSL_API(int,VSLGETBRNGPROPERTIES,(int *, VSLBRngProperties* ))


_Vsl_Api(int,vslSaveStreamF,(VSLStreamStatePtr ,  char*       ))
_vsl_api(int,vslsavestreamf,(VSLStreamStatePtr *, char* , int ))
_VSL_API(int,VSLSAVESTREAMF,(VSLStreamStatePtr *, char* , int ))

_Vsl_Api(int,vslLoadStreamF,(VSLStreamStatePtr *, char*       ))
_vsl_api(int,vslloadstreamf,(VSLStreamStatePtr *, char* , int ))
_VSL_API(int,VSLLOADSTREAMF,(VSLStreamStatePtr *, char* , int ))


/*
//++
//  VSL CONVOLUTION AND CORRELATION FUNCTION DECLARATIONS.
//--
*/

_Vsl_Api(int,vsldConvNewTask,(VSLConvTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT [], MKL_INT [], MKL_INT []));
_vsl_api(int,vsldconvnewtask,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT []));
_VSL_API(int,VSLDCONVNEWTASK,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT []));

_Vsl_Api(int,vslsConvNewTask,(VSLConvTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT [], MKL_INT [], MKL_INT []));
_vsl_api(int,vslsconvnewtask,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT []));
_VSL_API(int,VSLSCONVNEWTASK,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT []));

_Vsl_Api(int,vsldCorrNewTask,(VSLCorrTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT [], MKL_INT [], MKL_INT []));
_vsl_api(int,vsldcorrnewtask,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT []));
_VSL_API(int,VSLDCORRNEWTASK,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT []));

_Vsl_Api(int,vslsCorrNewTask,(VSLCorrTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT [], MKL_INT [], MKL_INT []));
_vsl_api(int,vslscorrnewtask,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT []));
_VSL_API(int,VSLSCORRNEWTASK,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT []));


_Vsl_Api(int,vsldConvNewTask1D,(VSLConvTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT ,  MKL_INT  ));
_vsl_api(int,vsldconvnewtask1d,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* ));
_VSL_API(int,VSLDCONVNEWTASK1D,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* ));

_Vsl_Api(int,vslsConvNewTask1D,(VSLConvTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT ,  MKL_INT  ));
_vsl_api(int,vslsconvnewtask1d,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* ));
_VSL_API(int,VSLSCONVNEWTASK1D,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* ));

_Vsl_Api(int,vsldCorrNewTask1D,(VSLCorrTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT ,  MKL_INT  ));
_vsl_api(int,vsldcorrnewtask1d,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* ));
_VSL_API(int,VSLDCORRNEWTASK1D,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* ));

_Vsl_Api(int,vslsCorrNewTask1D,(VSLCorrTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT ,  MKL_INT  ));
_vsl_api(int,vslscorrnewtask1d,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* ));
_VSL_API(int,VSLSCORRNEWTASK1D,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* ));


_Vsl_Api(int,vsldConvNewTaskX,(VSLConvTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT [], MKL_INT [], MKL_INT [], double [], MKL_INT []));
_vsl_api(int,vsldconvnewtaskx,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT [], double [], MKL_INT []));
_VSL_API(int,VSLDCONVNEWTASKX,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT [], double [], MKL_INT []));

_Vsl_Api(int,vslsConvNewTaskX,(VSLConvTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT [], MKL_INT [], MKL_INT [], float [],  MKL_INT []));
_vsl_api(int,vslsconvnewtaskx,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT [], float [],  MKL_INT []));
_VSL_API(int,VSLSCONVNEWTASKX,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT [], float [],  MKL_INT []));

_Vsl_Api(int,vsldCorrNewTaskX,(VSLCorrTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT [], MKL_INT [], MKL_INT [], double [], MKL_INT []));
_vsl_api(int,vsldcorrnewtaskx,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT [], double [], MKL_INT []));
_VSL_API(int,VSLDCORRNEWTASKX,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT [], double [], MKL_INT []));

_Vsl_Api(int,vslsCorrNewTaskX,(VSLCorrTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT [], MKL_INT [], MKL_INT [], float [],  MKL_INT []));
_vsl_api(int,vslscorrnewtaskx,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT [], float [],  MKL_INT []));
_VSL_API(int,VSLSCORRNEWTASKX,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT [], MKL_INT [], MKL_INT [], float [],  MKL_INT []));


_Vsl_Api(int,vsldConvNewTaskX1D,(VSLConvTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT  , MKL_INT  , double [], MKL_INT  ));
_vsl_api(int,vsldconvnewtaskx1d,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* , double [], MKL_INT* ));
_VSL_API(int,VSLDCONVNEWTASKX1D,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* , double [], MKL_INT* ));

_Vsl_Api(int,vslsConvNewTaskX1D,(VSLConvTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT  , MKL_INT  , float [],  MKL_INT  ));
_vsl_api(int,vslsconvnewtaskx1d,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* , float [],  MKL_INT* ));
_VSL_API(int,VSLSCONVNEWTASKX1D,(VSLConvTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* , float [],  MKL_INT* ));

_Vsl_Api(int,vsldCorrNewTaskX1D,(VSLCorrTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT  , MKL_INT  , double [], MKL_INT  ));
_vsl_api(int,vsldcorrnewtaskx1d,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* , double [], MKL_INT* ));
_VSL_API(int,VSLDCORRNEWTASKX1D,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* , double [], MKL_INT* ));

_Vsl_Api(int,vslsCorrNewTaskX1D,(VSLCorrTaskPtr* , MKL_INT  , MKL_INT  , MKL_INT  , MKL_INT  , float [],  MKL_INT  ));
_vsl_api(int,vslscorrnewtaskx1d,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* , float [],  MKL_INT* ));
_VSL_API(int,VSLSCORRNEWTASKX1D,(VSLCorrTaskPtr* , MKL_INT* , MKL_INT* , MKL_INT* , MKL_INT* , float [],  MKL_INT* ));


_Vsl_Api(int,vslConvDeleteTask,(VSLConvTaskPtr* ));
_vsl_api(int,vslconvdeletetask,(VSLConvTaskPtr* ));
_VSL_API(int,VSLCONVDeleteTask,(VSLConvTaskPtr* ));

_Vsl_Api(int,vslCorrDeleteTask,(VSLCorrTaskPtr* ));
_vsl_api(int,vslcorrdeletetask,(VSLCorrTaskPtr* ));
_VSL_API(int,VSLCORRDeleteTask,(VSLCorrTaskPtr* ));


_Vsl_Api(int,vslConvCopyTask,(VSLConvTaskPtr* , VSLConvTaskPtr ));
_vsl_api(int,vslconvcopytask,(VSLConvTaskPtr* , VSLConvTaskPtr ));
_VSL_API(int,VSLCONVCopyTask,(VSLConvTaskPtr* , VSLConvTaskPtr ));

_Vsl_Api(int,vslCorrCopyTask,(VSLCorrTaskPtr* , VSLCorrTaskPtr ));
_vsl_api(int,vslcorrcopytask,(VSLCorrTaskPtr* , VSLCorrTaskPtr ));
_VSL_API(int,VSLCORRCopyTask,(VSLCorrTaskPtr* , VSLCorrTaskPtr ));


_Vsl_Api(int,vslConvSetMode,(VSLConvTaskPtr , MKL_INT  ));
_vsl_api(int,vslconvsetmode,(VSLConvTaskPtr , MKL_INT* ));
_VSL_API(int,VSLCONVSETMOME,(VSLConvTaskPtr , MKL_INT* ));

_Vsl_Api(int,vslCorrSetMode,(VSLCorrTaskPtr , MKL_INT  ));
_vsl_api(int,vslcorrsetmode,(VSLCorrTaskPtr , MKL_INT* ));
_VSL_API(int,VSLCORRSETMODE,(VSLCorrTaskPtr , MKL_INT* ));


_Vsl_Api(int,vslConvSetInternalPrecision,(VSLConvTaskPtr , MKL_INT  ));
_vsl_api(int,vslconvsetinternalprecision,(VSLConvTaskPtr , MKL_INT* ));
_VSL_API(int,VSLCONVSETINTERNALPRECISION,(VSLConvTaskPtr , MKL_INT* ));

_Vsl_Api(int,vslCorrSetInternalPrecision,(VSLCorrTaskPtr , MKL_INT  ));
_vsl_api(int,vslcorrsetinternalprecision,(VSLCorrTaskPtr , MKL_INT* ));
_VSL_API(int,VSLCORRSETINTERNALPRECISION,(VSLCorrTaskPtr , MKL_INT* ));


_Vsl_Api(int,vslConvSetStart,(VSLConvTaskPtr , MKL_INT []));
_vsl_api(int,vslconvsetstart,(VSLConvTaskPtr , MKL_INT []));
_VSL_API(int,VSLCONVSETSTART,(VSLConvTaskPtr , MKL_INT []));

_Vsl_Api(int,vslCorrSetStart,(VSLCorrTaskPtr , MKL_INT []));
_vsl_api(int,vslcorrsetstart,(VSLCorrTaskPtr , MKL_INT []));
_VSL_API(int,VSLCORRSETSTART,(VSLCorrTaskPtr , MKL_INT []));


_Vsl_Api(int,vslConvSetDecimation,(VSLConvTaskPtr , MKL_INT []));
_vsl_api(int,vslconvsetdecimation,(VSLConvTaskPtr , MKL_INT []));
_VSL_API(int,VSLCONVSETDECIMATION,(VSLConvTaskPtr , MKL_INT []));

_Vsl_Api(int,vslCorrSetDecimation,(VSLCorrTaskPtr , MKL_INT []));
_vsl_api(int,vslcorrsetdecimation,(VSLCorrTaskPtr , MKL_INT []));
_VSL_API(int,VSLCORRSETDECIMATION,(VSLCorrTaskPtr , MKL_INT []));


_Vsl_Api(int,vsldConvExec,(VSLConvTaskPtr , double [], MKL_INT [], double [], MKL_INT [], double [], MKL_INT []));
_vsl_api(int,vsldconvexec,(VSLConvTaskPtr , double [], MKL_INT [], double [], MKL_INT [], double [], MKL_INT []));
_VSL_API(int,VSLDCONVEXEC,(VSLConvTaskPtr , double [], MKL_INT [], double [], MKL_INT [], double [], MKL_INT []));

_Vsl_Api(int,vslsConvExec,(VSLConvTaskPtr , float [],  MKL_INT [], float [],  MKL_INT [], float [],  MKL_INT []));
_vsl_api(int,vslsconvexec,(VSLConvTaskPtr , float [],  MKL_INT [], float [],  MKL_INT [], float [],  MKL_INT []));
_VSL_API(int,VSLSCONVEXEC,(VSLConvTaskPtr , float [],  MKL_INT [], float [],  MKL_INT [], float [],  MKL_INT []));

_Vsl_Api(int,vsldCorrExec,(VSLCorrTaskPtr , double [], MKL_INT [], double [], MKL_INT [], double [], MKL_INT []));
_vsl_api(int,vsldcorrexec,(VSLCorrTaskPtr , double [], MKL_INT [], double [], MKL_INT [], double [], MKL_INT []));
_VSL_API(int,VSLDCORREXEC,(VSLCorrTaskPtr , double [], MKL_INT [], double [], MKL_INT [], double [], MKL_INT []));

_Vsl_Api(int,vslsCorrExec,(VSLCorrTaskPtr , float [],  MKL_INT [], float [],  MKL_INT [], float [],  MKL_INT []));
_vsl_api(int,vslscorrexec,(VSLCorrTaskPtr , float [],  MKL_INT [], float [],  MKL_INT [], float [],  MKL_INT []));
_VSL_API(int,VSLSCORREXEC,(VSLCorrTaskPtr , float [],  MKL_INT [], float [],  MKL_INT [], float [],  MKL_INT []));


_Vsl_Api(int,vsldConvExec1D,(VSLConvTaskPtr , double [], MKL_INT  , double [], MKL_INT  , double [], MKL_INT  ));
_vsl_api(int,vsldconvexec1d,(VSLConvTaskPtr , double [], MKL_INT* , double [], MKL_INT* , double [], MKL_INT* ));
_VSL_API(int,VSLDCONVEXEC1D,(VSLConvTaskPtr , double [], MKL_INT* , double [], MKL_INT* , double [], MKL_INT* ));

_Vsl_Api(int,vslsConvExec1D,(VSLConvTaskPtr , float [],  MKL_INT  , float [],  MKL_INT  , float [],  MKL_INT  ));
_vsl_api(int,vslsconvexec1d,(VSLConvTaskPtr , float [],  MKL_INT* , float [],  MKL_INT* , float [],  MKL_INT* ));
_VSL_API(int,VSLSCONVEXEC1D,(VSLConvTaskPtr , float [],  MKL_INT* , float [],  MKL_INT* , float [],  MKL_INT* ));

_Vsl_Api(int,vsldCorrExec1D,(VSLCorrTaskPtr , double [], MKL_INT  , double [], MKL_INT  , double [], MKL_INT  ));
_vsl_api(int,vsldcorrexec1d,(VSLCorrTaskPtr , double [], MKL_INT* , double [], MKL_INT* , double [], MKL_INT* ));
_VSL_API(int,VSLDCORREXEC1D,(VSLCorrTaskPtr , double [], MKL_INT* , double [], MKL_INT* , double [], MKL_INT* ));

_Vsl_Api(int,vslsCorrExec1D,(VSLCorrTaskPtr , float [],  MKL_INT  , float [],  MKL_INT  , float [],  MKL_INT  ));
_vsl_api(int,vslscorrexec1d,(VSLCorrTaskPtr , float [],  MKL_INT* , float [],  MKL_INT* , float [],  MKL_INT* ));
_VSL_API(int,VSLSCORREXEC1D,(VSLCorrTaskPtr , float [],  MKL_INT* , float [],  MKL_INT* , float [],  MKL_INT* ));


_Vsl_Api(int,vsldConvExecX,(VSLConvTaskPtr , double [], MKL_INT [], double [], MKL_INT []));
_vsl_api(int,vsldconvexecx,(VSLConvTaskPtr , double [], MKL_INT [], double [], MKL_INT []));
_VSL_API(int,VSLDCONVEXECX,(VSLConvTaskPtr , double [], MKL_INT [], double [], MKL_INT []));

_Vsl_Api(int,vslsConvExecX,(VSLConvTaskPtr , float [],  MKL_INT [], float [],  MKL_INT []));
_vsl_api(int,vslsconvexecx,(VSLConvTaskPtr , float [],  MKL_INT [], float [],  MKL_INT []));
_VSL_API(int,VSLSCONVEXECX,(VSLConvTaskPtr , float [],  MKL_INT [], float [],  MKL_INT []));

_Vsl_Api(int,vsldCorrExecX,(VSLCorrTaskPtr , double [], MKL_INT [], double [], MKL_INT []));
_vsl_api(int,vsldcorrexecx,(VSLCorrTaskPtr , double [], MKL_INT [], double [], MKL_INT []));
_VSL_API(int,VSLDCORREXECX,(VSLCorrTaskPtr , double [], MKL_INT [], double [], MKL_INT []));

_Vsl_Api(int,vslsCorrExecX,(VSLCorrTaskPtr , float [],  MKL_INT [], float [],  MKL_INT []));
_vsl_api(int,vslscorrexecx,(VSLCorrTaskPtr , float [],  MKL_INT [], float [],  MKL_INT []));
_VSL_API(int,VSLSCORREXECX,(VSLCorrTaskPtr , float [],  MKL_INT [], float [],  MKL_INT []));


_Vsl_Api(int,vsldConvExecX1D,(VSLConvTaskPtr , double [], MKL_INT  , double [], MKL_INT  ));
_vsl_api(int,vsldconvexecx1d,(VSLConvTaskPtr , double [], MKL_INT* , double [], MKL_INT* ));
_VSL_API(int,VSLDCONVEXECX1D,(VSLConvTaskPtr , double [], MKL_INT* , double [], MKL_INT* ));

_Vsl_Api(int,vslsConvExecX1D,(VSLConvTaskPtr , float [],  MKL_INT  , float [],  MKL_INT  ));
_vsl_api(int,vslsconvexecx1d,(VSLConvTaskPtr , float [],  MKL_INT* , float [],  MKL_INT* ));
_VSL_API(int,VSLSCONVEXECX1D,(VSLConvTaskPtr , float [],  MKL_INT* , float [],  MKL_INT* ));

_Vsl_Api(int,vsldCorrExecX1D,(VSLCorrTaskPtr , double [], MKL_INT  , double [], MKL_INT  ));
_vsl_api(int,vsldcorrexecx1d,(VSLCorrTaskPtr , double [], MKL_INT* , double [], MKL_INT* ));
_VSL_API(int,VSLDCORREXECX1D,(VSLCorrTaskPtr , double [], MKL_INT* , double [], MKL_INT* ));

_Vsl_Api(int,vslsCorrExecX1D,(VSLCorrTaskPtr , float [],  MKL_INT  , float [],  MKL_INT  ));
_vsl_api(int,vslscorrexecx1d,(VSLCorrTaskPtr , float [],  MKL_INT* , float [],  MKL_INT* ));
_VSL_API(int,VSLSCORREXECX1D,(VSLCorrTaskPtr , float [],  MKL_INT* , float [],  MKL_INT* ));


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_VSL_FUNCTIONS_H__ */
