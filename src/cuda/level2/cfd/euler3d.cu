// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\cfd\euler3d.cu
//
// summary:	Sort class
// 
// origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

//#include <cutil.h>
#include <iostream>
#include <fstream>
#include "cudacommon.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Number streams. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define NUM_STREAMS 2

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines seed. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define SEED 7
 
/*
/// <summary>	. </summary>
 * Options 
 * 
 */ 
#define GAMMA 1.4f

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines iterations. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define iterations 10
// #ifndef block_length
// 	#define block_length 192
// #endif

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines ndim. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define NDIM 3

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines nnb. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define NNB 4

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines rk. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define RK 3	// 3rd order RK

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines ff mach. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define ff_mach 1.2f

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Degrees angle of attack. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define deg_angle_of_attack 0.0f

/*
 * not options
 */

#ifdef RD_WG_SIZE_0_0

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 0. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_0 RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 0. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_0 RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 0. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_0 RD_WG_SIZE
#else

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 0. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_0 192
#endif

#ifdef RD_WG_SIZE_1_0

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 1. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_1 RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 1. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_1 RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 1. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_1 RD_WG_SIZE
#else

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 1. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_1 192
#endif

#ifdef RD_WG_SIZE_2_0

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 2. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_2 RD_WG_SIZE_2_0
#elif defined(RD_WG_SIZE_1)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 2. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_2 RD_WG_SIZE_2
#elif defined(RD_WG_SIZE)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 2. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_2 RD_WG_SIZE
#else

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 2. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_2 192
#endif

#ifdef RD_WG_SIZE_3_0

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 3. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_3 RD_WG_SIZE_3_0
#elif defined(RD_WG_SIZE_3)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 3. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_3 RD_WG_SIZE_3
#elif defined(RD_WG_SIZE)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 3. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_3 RD_WG_SIZE
#else

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 3. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_3 192
#endif

#ifdef RD_WG_SIZE_4_0

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 4. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_4 RD_WG_SIZE_4_0
#elif defined(RD_WG_SIZE_4)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 4. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_4 RD_WG_SIZE_4
#elif defined(RD_WG_SIZE)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 4. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_4 RD_WG_SIZE
#else

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size 4. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define BLOCK_SIZE_4 192
#endif



// #if block_length > 128
// #warning "the kernels may fail too launch on some systems if the block length is too large"
// #endif

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Variable density. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define VAR_DENSITY 0

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Variable momentum. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define VAR_MOMENTUM  1

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Variable density energy. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines nvar. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define NVAR (VAR_DENSITY_ENERGY+1)

/// <summary>	The kernel time. </summary>
float kernelTime = 0.0f;
/// <summary>	The transfer time. </summary>
float transferTime = 0.0f;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the stop. </summary>
///
/// <value>	The stop. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

cudaEvent_t start, stop;
/// <summary>	The elapsed. </summary>
float elapsed;

/*

 ////////////////////////////////////////////////////////////////////////////////////////////////////
 /// <summary>	Allocs. </summary>
 ///
 /// <remarks>	Ed, 5/20/2020. </remarks>
 ///
 /// <typeparam name="T">	Generic type parameter. </typeparam>
 /// <param name="N">	An int to process. </param>
 ///
 /// <returns>	Null if it fails, else a pointer to a T. </returns>
 ////////////////////////////////////////////////////////////////////////////////////////////////////

 * Generic functions
 */
template <typename T>
T* alloc(int N)
{
	T* t;
	checkCudaErrors(cudaMalloc((void**)&t, sizeof(T)*N));
	return t;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Deallocs the given array. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="array">	[in,out] If non-null, the array. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void dealloc(T* array)
{
	checkCudaErrors(cudaFree((void*)array));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Copies this.  </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="dst">	[in,out] If non-null, destination for the. </param>
/// <param name="src">	[in,out] If non-null, source for the. </param>
/// <param name="N">  	An int to process. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void copy(T* dst, T* src, int N, cudaStream_t *stream)
{
    cudaEventRecord(start, 0);
	checkCudaErrors(cudaMemcpyAsync((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyDeviceToDevice, *stream));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    transferTime += elapsed * 1.e-3;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Uploads. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="dst">	[in,out] If non-null, destination for the. </param>
/// <param name="src">	[in,out] If non-null, source for the. </param>
/// <param name="N">  	An int to process. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void upload(T* dst, T* src, int N)
{
    cudaEventRecord(start, 0);
	checkCudaErrors(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyHostToDevice));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    transferTime += elapsed * 1.e-3;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Downloads this.  </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="dst">	[in,out] If non-null, destination for the. </param>
/// <param name="src">	[in,out] If non-null, source for the. </param>
/// <param name="N">  	An int to process. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void download(T* dst, T* src, int N)
{
    cudaEventRecord(start, 0);
	checkCudaErrors(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    transferTime += elapsed * 1.e-3;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Dumps. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="variables">	[in,out] If non-null, the variables. </param>
/// <param name="nel">			The nel. </param>
/// <param name="nelr">			The nelr. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void dump(float* variables, int nel, int nelr)
{
	float* h_variables = new float[nelr*NVAR];
	download(h_variables, variables, nelr*NVAR);

	{
		std::ofstream file("density");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY*nelr] << std::endl;
	}


	{
		std::ofstream file("momentum");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++)
		{
			for(int j = 0; j != NDIM; j++)
				file << h_variables[i + (VAR_MOMENTUM+j)*nelr] << " ";
			file << std::endl;
		}
	}
	
	{
		std::ofstream file("density_energy");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY_ENERGY*nelr] << std::endl;
	}
	delete[] h_variables;
}

/*

 ////////////////////////////////////////////////////////////////////////////////////////////////////
 /// <summary>	Gets the ff variable[ nvar]. </summary>
 ///
 /// <value>	The ff variable[ nvar]. </value>
 ////////////////////////////////////////////////////////////////////////////////////////////////////

 * Element-based Cell-centered FVM solver functions
 */
__constant__ float ff_variable[NVAR];

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the ff flux contribution momentum x[ 1]. </summary>
///
/// <value>	The ff flux contribution momentum x[ 1]. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

__constant__ float3 ff_flux_contribution_momentum_x[1];

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the ff flux contribution momentum y[ 1]. </summary>
///
/// <value>	The ff flux contribution momentum y[ 1]. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

__constant__ float3 ff_flux_contribution_momentum_y[1];

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the ff flux contribution momentum z[ 1]. </summary>
///
/// <value>	The ff flux contribution momentum z[ 1]. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

__constant__ float3 ff_flux_contribution_momentum_z[1];

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the ff flux contribution density energy[ 1]. </summary>
///
/// <value>	The ff flux contribution density energy[ 1]. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

__constant__ float3 ff_flux_contribution_density_energy[1];

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Cuda initialize variables. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="nelr">			The nelr. </param>
/// <param name="variables">	[in,out] If non-null, the variables. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void cuda_initialize_variables(int nelr, float* variables)
{
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);
	for(int j = 0; j < NVAR; j++)
		variables[i + j*nelr] = ff_variable[j];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Initializes the variables. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="nelr">			The nelr. </param>
/// <param name="variables">	[in,out] If non-null, the variables. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_variables(int nelr, float* variables)
{
	dim3 Dg(nelr / BLOCK_SIZE_1), Db(BLOCK_SIZE_1);
    cudaEventRecord(start, 0);
	cuda_initialize_variables<<<Dg, Db>>>(nelr, variables);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the flux contribution. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="density">				[in,out] The density. </param>
/// <param name="momentum">				[in,out] The momentum. </param>
/// <param name="density_energy">   	[in,out] The density energy. </param>
/// <param name="pressure">				[in,out] The pressure. </param>
/// <param name="velocity">				[in,out] The velocity. </param>
/// <param name="fc_momentum_x">		[in,out] The fc momentum x coordinate. </param>
/// <param name="fc_momentum_y">		[in,out] The fc momentum y coordinate. </param>
/// <param name="fc_momentum_z">		[in,out] The fc momentum z coordinate. </param>
/// <param name="fc_density_energy">	[in,out] The fc density energy. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __host__ inline void compute_flux_contribution(float& density, float3& momentum, float& density_energy, float& pressure, float3& velocity, float3& fc_momentum_x, float3& fc_momentum_y, float3& fc_momentum_z, float3& fc_density_energy)
{
	fc_momentum_x.x = velocity.x*momentum.x + pressure;
	fc_momentum_x.y = velocity.x*momentum.y;
	fc_momentum_x.z = velocity.x*momentum.z;
	
	
	fc_momentum_y.x = fc_momentum_x.y;
	fc_momentum_y.y = velocity.y*momentum.y + pressure;
	fc_momentum_y.z = velocity.y*momentum.z;

	fc_momentum_z.x = fc_momentum_x.z;
	fc_momentum_z.y = fc_momentum_y.z;
	fc_momentum_z.z = velocity.z*momentum.z + pressure;

	float de_p = density_energy+pressure;
	fc_density_energy.x = velocity.x*de_p;
	fc_density_energy.y = velocity.y*de_p;
	fc_density_energy.z = velocity.z*de_p;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the velocity. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="density"> 	[in,out] The density. </param>
/// <param name="momentum">	[in,out] The momentum. </param>
/// <param name="velocity">	[in,out] The velocity. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ inline void compute_velocity(float& density, float3& momentum, float3& velocity)
{
	velocity.x = momentum.x / density;
	velocity.y = momentum.y / density;
	velocity.z = momentum.z / density;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the speed sqd. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="velocity">	[in,out] The velocity. </param>
///
/// <returns>	The calculated speed sqd. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ inline float compute_speed_sqd(float3& velocity)
{
	return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the pressure. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="density">		 	[in,out] The density. </param>
/// <param name="density_energy">	[in,out] The density energy. </param>
/// <param name="speed_sqd">	 	[in,out] The speed sqd. </param>
///
/// <returns>	The calculated pressure. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ inline float compute_pressure(float& density, float& density_energy, float& speed_sqd)
{
	return (float(GAMMA)-float(1.0f))*(density_energy - float(0.5f)*density*speed_sqd);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the speed of sound. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="density"> 	[in,out] The density. </param>
/// <param name="pressure">	[in,out] The pressure. </param>
///
/// <returns>	The calculated speed of sound. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ inline float compute_speed_of_sound(float& density, float& pressure)
{
	return sqrtf(float(GAMMA)*pressure/density);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Cuda compute step factor. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="nelr">		   	The nelr. </param>
/// <param name="variables">   	[in,out] If non-null, the variables. </param>
/// <param name="areas">	   	[in,out] If non-null, the areas. </param>
/// <param name="step_factors">	[in,out] If non-null, the step factors. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void cuda_compute_step_factor(int nelr, float* variables, float* areas, float* step_factors)
{
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);

	float density = variables[i + VAR_DENSITY*nelr];
	float3 momentum;
	momentum.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	momentum.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	momentum.z = variables[i + (VAR_MOMENTUM+2)*nelr];
	
	float density_energy = variables[i + VAR_DENSITY_ENERGY*nelr];
	
	float3 velocity;       compute_velocity(density, momentum, velocity);
	float speed_sqd      = compute_speed_sqd(velocity);
	float pressure       = compute_pressure(density, density_energy, speed_sqd);
	float speed_of_sound = compute_speed_of_sound(density, pressure);

	// dt = float(0.5f) * sqrtf(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
	step_factors[i] = float(0.5f) / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
}

#ifdef HYPERQ

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the step factor. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="nelr">		   	The nelr. </param>
/// <param name="variables">   	[in,out] If non-null, the variables. </param>
/// <param name="areas">	   	[in,out] If non-null, the areas. </param>
/// <param name="step_factors">	[in,out] If non-null, the step factors. </param>
/// <param name="stream">	   	[in,out] If non-null, the stream. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void compute_step_factor(int nelr, float* variables, float* areas, float* step_factors, cudaStream_t *stream)
{
	dim3 Dg(nelr / BLOCK_SIZE_2), Db(BLOCK_SIZE_2);
    cudaEventRecord(start, 0);
	cuda_compute_step_factor<<<Dg, Db, 0, *stream>>>(nelr, variables, areas, step_factors);		
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the step factor. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="nelr">		   	The nelr. </param>
/// <param name="variables">   	[in,out] If non-null, the variables. </param>
/// <param name="areas">	   	[in,out] If non-null, the areas. </param>
/// <param name="step_factors">	[in,out] If non-null, the step factors. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void compute_step_factor(int nelr, float* variables, float* areas, float* step_factors, cudaStream_t *stream)
{
	dim3 Dg(nelr / BLOCK_SIZE_2), Db(BLOCK_SIZE_2);
    cudaEventRecord(start, 0);
	cuda_compute_step_factor<<<Dg, Db, 0, *stream>>>(nelr, variables, areas, step_factors);		
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Cuda compute flux. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="nelr">								The nelr. </param>
/// <param name="elements_surrounding_elements">	[in,out] If non-null, the elements
/// 												surrounding elements. </param>
/// <param name="normals">							[in,out] If non-null, the normals. </param>
/// <param name="variables">						[in,out] If non-null, the variables. </param>
/// <param name="fluxes">							[in,out] If non-null, the fluxes. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void cuda_compute_flux(int nelr, int* elements_surrounding_elements, float* normals, float* variables, float* fluxes)
{
	const float smoothing_coefficient = float(0.2f);
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);
	
	int j, nb;
	float3 normal; float normal_len;
	float factor;
	
	float density_i = variables[i + VAR_DENSITY*nelr];
	float3 momentum_i;
	momentum_i.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	momentum_i.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	momentum_i.z = variables[i + (VAR_MOMENTUM+2)*nelr];

	float density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];

	float3 velocity_i;             				compute_velocity(density_i, momentum_i, velocity_i);
	float speed_sqd_i                          = compute_speed_sqd(velocity_i);
	float speed_i                              = sqrtf(speed_sqd_i);
	float pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
	float speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
	float3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
	float3 flux_contribution_i_density_energy;	
	compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z, flux_contribution_i_density_energy);
	
	float flux_i_density = float(0.0f);
	float3 flux_i_momentum;
	flux_i_momentum.x = float(0.0f);
	flux_i_momentum.y = float(0.0f);
	flux_i_momentum.z = float(0.0f);
	float flux_i_density_energy = float(0.0f);
		
	float3 velocity_nb;
	float density_nb, density_energy_nb;
	float3 momentum_nb;
	float3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
	float3 flux_contribution_nb_density_energy;	
	float speed_sqd_nb, speed_of_sound_nb, pressure_nb;
	
	#pragma unroll
	for(j = 0; j < NNB; j++)
	{
		nb = elements_surrounding_elements[i + j*nelr];
		normal.x = normals[i + (j + 0*NNB)*nelr];
		normal.y = normals[i + (j + 1*NNB)*nelr];
		normal.z = normals[i + (j + 2*NNB)*nelr];
		normal_len = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
		
		if(nb >= 0) 	// a legitimate neighbor
		{
			density_nb = variables[nb + VAR_DENSITY*nelr];
			momentum_nb.x = variables[nb + (VAR_MOMENTUM+0)*nelr];
			momentum_nb.y = variables[nb + (VAR_MOMENTUM+1)*nelr];
			momentum_nb.z = variables[nb + (VAR_MOMENTUM+2)*nelr];
			density_energy_nb = variables[nb + VAR_DENSITY_ENERGY*nelr];
												compute_velocity(density_nb, momentum_nb, velocity_nb);
			speed_sqd_nb                      = compute_speed_sqd(velocity_nb);
			pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
			speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);
			                                    compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z, flux_contribution_nb_density_energy);
			
			// artificial viscosity
			factor = -normal_len*smoothing_coefficient*float(0.5f)*(speed_i + sqrtf(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
			flux_i_density += factor*(density_i-density_nb);
			flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
			flux_i_momentum.x += factor*(momentum_i.x-momentum_nb.x);
			flux_i_momentum.y += factor*(momentum_i.y-momentum_nb.y);
			flux_i_momentum.z += factor*(momentum_i.z-momentum_nb.z);

			// accumulate cell-centered fluxes
			factor = float(0.5f)*normal.x;
			flux_i_density += factor*(momentum_nb.x+momentum_i.x);
			flux_i_density_energy += factor*(flux_contribution_nb_density_energy.x+flux_contribution_i_density_energy.x);
			flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.x+flux_contribution_i_momentum_x.x);
			flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.x+flux_contribution_i_momentum_y.x);
			flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.x+flux_contribution_i_momentum_z.x);
			
			factor = float(0.5f)*normal.y;
			flux_i_density += factor*(momentum_nb.y+momentum_i.y);
			flux_i_density_energy += factor*(flux_contribution_nb_density_energy.y+flux_contribution_i_density_energy.y);
			flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.y+flux_contribution_i_momentum_x.y);
			flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.y+flux_contribution_i_momentum_y.y);
			flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.y+flux_contribution_i_momentum_z.y);
			
			factor = float(0.5f)*normal.z;
			flux_i_density += factor*(momentum_nb.z+momentum_i.z);
			flux_i_density_energy += factor*(flux_contribution_nb_density_energy.z+flux_contribution_i_density_energy.z);
			flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.z+flux_contribution_i_momentum_x.z);
			flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.z+flux_contribution_i_momentum_y.z);
			flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.z+flux_contribution_i_momentum_z.z);
		}
		else if(nb == -1)	// a wing boundary
		{
			flux_i_momentum.x += normal.x*pressure_i;
			flux_i_momentum.y += normal.y*pressure_i;
			flux_i_momentum.z += normal.z*pressure_i;
		}
		else if(nb == -2) // a far field boundary
		{
			factor = float(0.5f)*normal.x;
			flux_i_density += factor*(ff_variable[VAR_MOMENTUM+0]+momentum_i.x);
			flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].x+flux_contribution_i_density_energy.x);
			flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].x + flux_contribution_i_momentum_x.x);
			flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].x + flux_contribution_i_momentum_y.x);
			flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].x + flux_contribution_i_momentum_z.x);
			
			factor = float(0.5f)*normal.y;
			flux_i_density += factor*(ff_variable[VAR_MOMENTUM+1]+momentum_i.y);
			flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].y+flux_contribution_i_density_energy.y);
			flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].y + flux_contribution_i_momentum_x.y);
			flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].y + flux_contribution_i_momentum_y.y);
			flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].y + flux_contribution_i_momentum_z.y);

			factor = float(0.5f)*normal.z;
			flux_i_density += factor*(ff_variable[VAR_MOMENTUM+2]+momentum_i.z);
			flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].z+flux_contribution_i_density_energy.z);
			flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].z + flux_contribution_i_momentum_x.z);
			flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].z + flux_contribution_i_momentum_y.z);
			flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].z + flux_contribution_i_momentum_z.z);

		}
	}

	fluxes[i + VAR_DENSITY*nelr] = flux_i_density;
	fluxes[i + (VAR_MOMENTUM+0)*nelr] = flux_i_momentum.x;
	fluxes[i + (VAR_MOMENTUM+1)*nelr] = flux_i_momentum.y;
	fluxes[i + (VAR_MOMENTUM+2)*nelr] = flux_i_momentum.z;
	fluxes[i + VAR_DENSITY_ENERGY*nelr] = flux_i_density_energy;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Calculates the flux. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="nelr">								The nelr. </param>
/// <param name="elements_surrounding_elements">	[in,out] If non-null, the elements
/// 												surrounding elements. </param>
/// <param name="normals">							[in,out] If non-null, the normals. </param>
/// <param name="variables">						[in,out] If non-null, the variables. </param>
/// <param name="fluxes">							[in,out] If non-null, the fluxes. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void compute_flux(int nelr, int* elements_surrounding_elements, float* normals, float* variables, float* fluxes, cudaStream_t *stream)
{
	dim3 Dg(nelr / BLOCK_SIZE_3), Db(BLOCK_SIZE_3);
    cudaEventRecord(start, 0);
	cuda_compute_flux<<<Dg,Db,0, *stream>>>(nelr, elements_surrounding_elements, normals, variables, fluxes);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Cuda time step. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="j">				An int to process. </param>
/// <param name="nelr">				The nelr. </param>
/// <param name="old_variables">	[in,out] If non-null, the old variables. </param>
/// <param name="variables">		[in,out] If non-null, the variables. </param>
/// <param name="step_factors"> 	[in,out] If non-null, the step factors. </param>
/// <param name="fluxes">			[in,out] If non-null, the fluxes. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void cuda_time_step(int j, int nelr, float* old_variables, float* variables, float* step_factors, float* fluxes)
{
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);

	float factor = step_factors[i]/float(RK+1-j);

	variables[i + VAR_DENSITY*nelr] = old_variables[i + VAR_DENSITY*nelr] + factor*fluxes[i + VAR_DENSITY*nelr];
	variables[i + VAR_DENSITY_ENERGY*nelr] = old_variables[i + VAR_DENSITY_ENERGY*nelr] + factor*fluxes[i + VAR_DENSITY_ENERGY*nelr];
	variables[i + (VAR_MOMENTUM+0)*nelr] = old_variables[i + (VAR_MOMENTUM+0)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+0)*nelr];
	variables[i + (VAR_MOMENTUM+1)*nelr] = old_variables[i + (VAR_MOMENTUM+1)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+1)*nelr];	
	variables[i + (VAR_MOMENTUM+2)*nelr] = old_variables[i + (VAR_MOMENTUM+2)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+2)*nelr];	
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Time step. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="j">				An int to process. </param>
/// <param name="nelr">				The nelr. </param>
/// <param name="old_variables">	[in,out] If non-null, the old variables. </param>
/// <param name="variables">		[in,out] If non-null, the variables. </param>
/// <param name="step_factors"> 	[in,out] If non-null, the step factors. </param>
/// <param name="fluxes">			[in,out] If non-null, the fluxes. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void time_step(int j, int nelr, float* old_variables, float* variables, float* step_factors, float* fluxes, cudaStream_t *stream)
{
	dim3 Dg(nelr / BLOCK_SIZE_4), Db(BLOCK_SIZE_4);
    cudaEventRecord(start, 0);
	cuda_time_step<<<Dg,Db,0,*stream>>>(j, nelr, old_variables, variables, step_factors, fluxes);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds a benchmark specifier options. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="op">	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void addBenchmarkSpecOptions(OptionParser &op) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Cfds. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void cfd(ResultDatabase &resultDB, OptionParser &op);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the benchmark operation. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
    printf("Running CFDSolver\n");
    bool quiet = op.getOptionBool("quiet");
    if(!quiet) {
    printf("WG size of kernel:initialize = %d, WG size of kernel:compute_step_factor = %d, WG size of kernel:compute_flux = %d, WG size of kernel:time_step = %d\n", BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3, BLOCK_SIZE_4);
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int passes = op.getOptionInt("passes");
    for(int i = 0; i < passes; i++) {
        kernelTime = 0.0f;
        transferTime = 0.0f;
        if(!quiet) {
            printf("Pass %d:\n", i);
        }
        cfd(resultDB, op);
        if(!quiet) {
            printf("Done.\n");
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Cfds. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void cfd(ResultDatabase &resultDB, OptionParser &op)
{
	// set far field conditions and load them into constant memory on the gpu
	{
		float h_ff_variable[NVAR];
		const float angle_of_attack = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);
		
		h_ff_variable[VAR_DENSITY] = float(1.4);
		
		float ff_pressure = float(1.0f);
		float ff_speed_of_sound = sqrt(GAMMA*ff_pressure / h_ff_variable[VAR_DENSITY]);
		float ff_speed = float(ff_mach)*ff_speed_of_sound;
		
		float3 ff_velocity;
		ff_velocity.x = ff_speed*float(cos((float)angle_of_attack));
		ff_velocity.y = ff_speed*float(sin((float)angle_of_attack));
		ff_velocity.z = 0.0f;
		
		h_ff_variable[VAR_MOMENTUM+0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
		h_ff_variable[VAR_MOMENTUM+1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
		h_ff_variable[VAR_MOMENTUM+2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;
				
		h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY]*(float(0.5f)*(ff_speed*ff_speed)) + (ff_pressure / float(GAMMA-1.0f));

		float3 h_ff_momentum;
		h_ff_momentum.x = *(h_ff_variable+VAR_MOMENTUM+0);
		h_ff_momentum.y = *(h_ff_variable+VAR_MOMENTUM+1);
		h_ff_momentum.z = *(h_ff_variable+VAR_MOMENTUM+2);
		float3 h_ff_flux_contribution_momentum_x;
		float3 h_ff_flux_contribution_momentum_y;
		float3 h_ff_flux_contribution_momentum_z;
		float3 h_ff_flux_contribution_density_energy;
		compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum, h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity, h_ff_flux_contribution_momentum_x, h_ff_flux_contribution_momentum_y, h_ff_flux_contribution_momentum_z, h_ff_flux_contribution_density_energy);

		// copy far field conditions to the gpu
        cudaEventRecord(start, 0);

		checkCudaErrors( cudaMemcpyToSymbol(ff_variable,          h_ff_variable,          NVAR*sizeof(float)) );
		checkCudaErrors( cudaMemcpyToSymbol(ff_flux_contribution_momentum_x, &h_ff_flux_contribution_momentum_x, sizeof(float3)) );
		checkCudaErrors( cudaMemcpyToSymbol(ff_flux_contribution_momentum_y, &h_ff_flux_contribution_momentum_y, sizeof(float3)) );
		checkCudaErrors( cudaMemcpyToSymbol(ff_flux_contribution_momentum_z, &h_ff_flux_contribution_momentum_z, sizeof(float3)) );
		checkCudaErrors( cudaMemcpyToSymbol(ff_flux_contribution_density_energy, &h_ff_flux_contribution_density_energy, sizeof(float3)) );		

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        transferTime += elapsed * 1.e-3;
	}

	bool uvm = op.getOptionBool("uvm");
	int nel;
	int nelr;
	
	// read in domain geometry
	float* areas;
	int* elements_surrounding_elements;
	float* normals;
	{
        string inputFile = op.getOptionString("inputFile");
		std::ifstream file(inputFile.c_str());
        
        if(inputFile != "") {
		    file >> nel;
        } else {
            int problemSizes[4] = {97000, 200000, 40000000, 60000000};
            nel = problemSizes[op.getOptionInt("size") - 1];
        }
		nelr = BLOCK_SIZE_0*((nel / BLOCK_SIZE_0 )+ std::min(1, nel % BLOCK_SIZE_0));

		float *h_areas = NULL;
		int *h_elements_surrounding_elements = NULL;
		float *h_normals = NULL;
		if (uvm) {
			// could use prefetch and advise
			checkCudaErrors(cudaMallocManaged(&h_areas, nelr * sizeof(float)));
			checkCudaErrors(cudaMallocManaged(&h_elements_surrounding_elements,
									nelr * NNB * sizeof(int)));
			checkCudaErrors(cudaMallocManaged(&h_normals, nelr * NDIM * NNB * sizeof(float)));
		} else {
			h_areas = new float[nelr];
			h_elements_surrounding_elements = new int[nelr*NNB];
			h_normals = new float[nelr*NDIM*NNB];
		}

		srand(SEED);
				
		// read in data
		for(int i = 0; i < nel; i++)
		{
            if(inputFile != "") {
		    	file >> h_areas[i];
            } else {
                h_areas[i] = 1.0 * rand() / RAND_MAX;
            }
			for(int j = 0; j < NNB; j++) // NNB is always 4
			{
                if(inputFile != "") {
				    file >> h_elements_surrounding_elements[i + j*nelr];
                } else {
                    int val = i + (rand() % 20) - 10;
                    h_elements_surrounding_elements[i + j * nelr] = val;
                }
				if(h_elements_surrounding_elements[i+j*nelr] < 0) h_elements_surrounding_elements[i+j*nelr] = -1;
				h_elements_surrounding_elements[i + j*nelr]--; //it's coming in with Fortran numbering				
				
				for(int k = 0; k < NDIM; k++) // NDIM is always 3
				{
                    if(inputFile != "") {
					    file >> h_normals[i + (j + k*NNB)*nelr];
                    } else {
                        h_normals[i + (j + k*NNB)*nelr] = 1.0 * rand() / RAND_MAX - 0.5;
                    }
					h_normals[i + (j + k*NNB)*nelr] = -h_normals[i + (j + k*NNB)*nelr];
				}
			}
		}
		
		// fill in remaining data
		int last = nel-1;
		for(int i = nel; i < nelr; i++)
		{
			h_areas[i] = h_areas[last];
			for(int j = 0; j < NNB; j++)
			{
				// duplicate the last element
				h_elements_surrounding_elements[i + j*nelr] = h_elements_surrounding_elements[last + j*nelr];	
				for(int k = 0; k < NDIM; k++) h_normals[last + (j + k*NNB)*nelr] = h_normals[last + (j + k*NNB)*nelr];
			}
		}
		
		if (uvm) {
			areas = h_areas;
			elements_surrounding_elements = h_elements_surrounding_elements;
			normals = h_normals;
		} else {
			areas = alloc<float>(nelr);
			upload<float>(areas, h_areas, nelr);

			elements_surrounding_elements = alloc<int>(nelr*NNB);
			upload<int>(elements_surrounding_elements, h_elements_surrounding_elements, nelr*NNB);

			normals = alloc<float>(nelr*NDIM*NNB);
			upload<float>(normals, h_normals, nelr*NDIM*NNB);
					
			delete[] h_areas;
			delete[] h_elements_surrounding_elements;
			delete[] h_normals;
		}
	}

	// Create arrays and set initial conditions
	float *variables = NULL;
	if (uvm) {
		checkCudaErrors(cudaMallocManaged(&variables, nelr*NVAR*sizeof(float)));
	} else {
		variables = alloc<float>(nelr*NVAR);
	}

	initialize_variables(nelr, variables);

	float *old_variables = NULL;
	float *fluxes = NULL;
	float *step_factors = NULL;
	if (uvm) {
		checkCudaErrors(cudaMallocManaged(&old_variables, nelr*NVAR*sizeof(float)));
		checkCudaErrors(cudaMallocManaged(&fluxes, nelr*NVAR*sizeof(float)));
		checkCudaErrors(cudaMallocManaged(&step_factors, nelr*sizeof(float)));
	} else {
		old_variables = alloc<float>(nelr*NVAR);   	
		fluxes = alloc<float>(nelr*NVAR);
		step_factors = alloc<float>(nelr);
	}

	// make sure all memory is floatly allocated before we start timing
	initialize_variables(nelr, old_variables);
	initialize_variables(nelr, fluxes);
	checkCudaErrors(cudaMemset( (void*) step_factors, 0, sizeof(float)*nelr ));
	// make sure CUDA isn't still doing something before we start timing
	checkCudaErrors(cudaDeviceSynchronize());

	// these need to be computed the first time in order to compute time step

	  //	unsigned int timer = 0;

	// CUT_SAFE_CALL( cutCreateTimer( &timer));
	// CUT_SAFE_CALL( cutStartTimer( timer));
	// Begin iterations

	// Only 2 streams here, may change later
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        checkCudaErrors(cudaStreamCreate(&streams[s]));
    }

	cudaEvent_t copy_event, exec_event;
	checkCudaErrors(cudaEventCreate(&copy_event));
	checkCudaErrors(cudaEventCreate(&exec_event));
	checkCudaErrors(cudaEventRecord(exec_event, streams[0]));
	for (int i = 0; i < iterations; i++) {
        // Time will need to be recomputed, more aggressive optimization TODO
		checkCudaErrors(cudaStreamWaitEvent(streams[1], exec_event, 0));
		copy<float>(old_variables, variables, nelr*NVAR, &streams[1]);
		checkCudaErrors(cudaEventRecord(copy_event, streams[1]));
	
		// for the first iteration we compute the time step
		compute_step_factor(nelr, variables, areas, step_factors, &streams[0]);
        CHECK_CUDA_ERROR();
		checkCudaErrors(cudaStreamWaitEvent(streams[0], copy_event, 0));
		
		for(int j = 0; j < RK; j++)
		{
			compute_flux(nelr, elements_surrounding_elements, normals, variables, fluxes, &streams[0]);
            CHECK_CUDA_ERROR(); 
			time_step(j, nelr, old_variables, variables, step_factors, fluxes, &streams[0]);
            CHECK_CUDA_ERROR();
		}
		checkCudaErrors(cudaEventRecord(exec_event, streams[0]));
	}
	checkCudaErrors(cudaEventDestroy(exec_event));
	checkCudaErrors(cudaEventDestroy(copy_event));

	checkCudaErrors(cudaDeviceSynchronize());

    if (op.getOptionBool("verbose")) {
	    dump(variables, nel, nelr);
    }

	if (uvm) {
		checkCudaErrors(cudaFree(areas));
		checkCudaErrors(cudaFree(elements_surrounding_elements));
		checkCudaErrors(cudaFree(normals));
		checkCudaErrors(cudaFree(variables));
		checkCudaErrors(cudaFree(old_variables));
		checkCudaErrors(cudaFree(fluxes));
		checkCudaErrors(cudaFree(step_factors));
	} else {
		dealloc<float>(areas);
		dealloc<int>(elements_surrounding_elements);
		dealloc<float>(normals);
		
		dealloc<float>(variables);
		dealloc<float>(old_variables);
		dealloc<float>(fluxes);
		dealloc<float>(step_factors);
	}
	
	for (int s = 0; s < NUM_STREAMS; s++) {
        checkCudaErrors(cudaStreamDestroy(streams[s]));
    }

	char atts[1024];
    sprintf(atts, "numelements:%d", nel);
    resultDB.AddResult("cfd_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("cfd_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("cfd_parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime+transferTime);
}
