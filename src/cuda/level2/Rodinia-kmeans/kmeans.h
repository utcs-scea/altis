/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/ 
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/

#ifndef _H_FUZZY_KMEANS
/// <summary>	. </summary>
#define _H_FUZZY_KMEANS

#ifndef FLT_MAX

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Filter Maximum. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define FLT_MAX 3.40282347e+38
#endif

#include "ResultDatabase.h"
#include "cudacommon.h"

/* rmse.c */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Euclid distance. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
///
/// <param name="parameter1">	[in,out] If non-null, the first parameter. </param>
/// <param name="parameter2">	[in,out] If non-null, the second parameter. </param>
/// <param name="parameter3">	The third parameter. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float   euclid_dist_2        (float*, float*, int);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Searches for the nearest point. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
///
/// <param name="parameter1">	[in,out] If non-null, the first parameter. </param>
/// <param name="parameter2">	The second parameter. </param>
/// <param name="parameter3">	[in,out] If non-null, the third parameter. </param>
/// <param name="parameter4">	The fourth parameter. </param>
///
/// <returns>	The found point. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

int     find_nearest_point   (float* , int, float**, int);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Rms error. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
///
/// <param name="parameter1">	[in,out] If non-null, the first parameter. </param>
/// <param name="parameter2">	The second parameter. </param>
/// <param name="parameter3">	The third parameter. </param>
/// <param name="parameter4">	[in,out] If non-null, the fourth parameter. </param>
/// <param name="parameter5">	The fifth parameter. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float	rms_err(float**, int, int, float**, int);

/* cluster.c */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Clusters. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
///
/// <param name="parameter1"> 	The first parameter. </param>
/// <param name="parameter2"> 	The second parameter. </param>
/// <param name="parameter3"> 	[in,out] If non-null, the third parameter. </param>
/// <param name="parameter4"> 	The fourth parameter. </param>
/// <param name="parameter5"> 	The fifth parameter. </param>
/// <param name="parameter6"> 	The parameter 6. </param>
/// <param name="parameter7"> 	[in,out] If non-null, the parameter 7. </param>
/// <param name="parameter8"> 	[in,out] If non-null, the parameter 8. </param>
/// <param name="parameter9"> 	[in,out] If non-null, the parameter 9. </param>
/// <param name="parameter10">	The parameter 10. </param>
/// <param name="parameter11">	The parameter 11. </param>
/// <param name="parameter12">	[in,out] The parameter 12. </param>
/// <param name="parameter13">	True to parameter 13. </param>
///
/// <returns>	An int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

int     cluster(int, int, float**, int, int, float, int*, float***, float*, int, int, int, ResultDatabase&, bool);

/* kmeans_clustering.c */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Kmeans clustering. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu) 5/20/2020. </remarks>
///
/// <param name="parameter1">	[in,out] If non-null, the first parameter. </param>
/// <param name="parameter2">	The second parameter. </param>
/// <param name="parameter3">	The third parameter. </param>
/// <param name="parameter4">	The fourth parameter. </param>
/// <param name="parameter5">	The fifth parameter. </param>
/// <param name="parameter6">	[in,out] If non-null, the parameter 6. </param>
/// <param name="parameter7">	[in,out] The parameter 7. </param>
/// <param name="parameter8">	True to parameter 8. </param>
///
/// <returns>	Null if it fails, else a handle to a float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float **kmeans_clustering(float**, int, int, int, int, float, int*, ResultDatabase&, bool);

#endif
