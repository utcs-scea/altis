/* 
 * Copyright (c) 2009, Jiri Matela
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // file:	altis\src\cuda\level2\dwt2d\dwt_cuda\dwt.h
 //
 // summary:	Sort class
 // 
 // origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
 ////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _DWT_H
/// <summary>	. </summary>
#define _DWT_H

#ifdef HYPERQ

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Stage 2D dwt with HyperQ. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="in">		   	[in,out] If non-null, the in. </param>
/// <param name="out">		   	[in,out] If non-null, the out. </param>
/// <param name="backup">	   	[in,out] If non-null, the backup. </param>
/// <param name="pixWidth">	   	Width of the pix. </param>
/// <param name="pixHeight">   	Height of the pix. </param>
/// <param name="stages">	   	The stages. </param>
/// <param name="forward">	   	True to forward. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
/// <param name="verbose">	   	True to verbose. </param>
/// <param name="quiet">	   	True to quiet. </param>
/// <param name="stream">	   	The stream. </param>
///
/// <returns>	An int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T> 
int nStage2dDWT(T *in, T *out, T * backup, int pixWidth, int pixHeight, int stages, bool forward, float &transferTime, float &kernelTime, bool verbose, bool quiet, cudaStream_t stream);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Stage 2D dwt. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="in">		   	[in,out] If non-null, the in. </param>
/// <param name="out">		   	[in,out] If non-null, the out. </param>
/// <param name="backup">	   	[in,out] If non-null, the backup. </param>
/// <param name="pixWidth">	   	Width of the pix. </param>
/// <param name="pixHeight">   	Height of the pix. </param>
/// <param name="stages">	   	The stages. </param>
/// <param name="forward">	   	True to forward. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
/// <param name="verbose">	   	True to verbose. </param>
/// <param name="quiet">	   	True to quiet. </param>
///
/// <returns>	An int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T> 
int nStage2dDWT(T *in, T *out, T * backup, int pixWidth, int pixHeight, int stages, bool forward, float &transferTime, float &kernelTime, bool verbose, bool quiet);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes a n stage 2D dwt. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="component_cuda">	[in,out] If non-null, the component cuda. </param>
/// <param name="width">		 	The width. </param>
/// <param name="height">		 	The height. </param>
/// <param name="stages">		 	The stages. </param>
/// <param name="filename">		 	Filename of the file. </param>
/// <param name="suffix">		 	The suffix. </param>
///
/// <returns>	An int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
int writeNStage2DDWT(T *component_cuda, int width, int height, 
                     int stages, const char * filename, const char * suffix);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Writes a linear. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="component_cuda">	[in,out] If non-null, the component cuda. </param>
/// <param name="width">		 	The width. </param>
/// <param name="height">		 	The height. </param>
/// <param name="filename">		 	Filename of the file. </param>
/// <param name="suffix">		 	The suffix. </param>
///
/// <returns>	An int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
int writeLinear(T *component_cuda, int width, int height, 
                     const char * filename, const char * suffix);

#endif
