// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\cfd\euler3d_double.cu
//
// summary:	Sort class
// 
// origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////


// #include <cutil.h>
#include <iostream>
#include <fstream>
#include "cudacommon.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines seed. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define SEED 7

#if CUDART_VERSION < 3000

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A double 3. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

struct double3
{
	////////////////////////////////////////////////////////////////////////////////////////////////////
	/// <summary>	Gets the z coordinate. </summary>
	///
	/// <value>	The z coordinate. </value>
	////////////////////////////////////////////////////////////////////////////////////////////////////

	double x, y, z;
};
#endif

/*
/// <summary>	. </summary>
 * Options 
 * 
 */ 
#define GAMMA 1.4

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines iterations. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define iterations 2000
#ifndef block_length

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block length. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

	#define block_length 128
#endif

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

#define ff_mach 1.2

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines Degrees angle of attack. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define deg_angle_of_attack 0.0

/*
 * not options
 */


#if block_length > 128
#warning "the kernels may fail too launch on some systems if the block length is too large"
#endif

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
	CUDA_SAFE_CALL(cudaMalloc((void**)&t, sizeof(T)*N));
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
	CUDA_SAFE_CALL(cudaFree((void*)array));
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
void copy(T* dst, T* src, int N)
{
    cudaEventRecord(start, 0);
	CUDA_SAFE_CALL(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyDeviceToDevice));
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
	CUDA_SAFE_CALL(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyHostToDevice));
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
	CUDA_SAFE_CALL(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyDeviceToHost));
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

void dump(double* variables, int nel, int nelr)
{
	double* h_variables = new double[nelr*NVAR];
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
__constant__ double ff_variable[NVAR];

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the ff flux contribution momentum x[ 1]. </summary>
///
/// <value>	The ff flux contribution momentum x[ 1]. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

__constant__ double3 ff_flux_contribution_momentum_x[1];

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the ff flux contribution momentum y[ 1]. </summary>
///
/// <value>	The ff flux contribution momentum y[ 1]. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

__constant__ double3 ff_flux_contribution_momentum_y[1];

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the ff flux contribution momentum z[ 1]. </summary>
///
/// <value>	The ff flux contribution momentum z[ 1]. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

__constant__ double3 ff_flux_contribution_momentum_z[1];

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gets the ff flux contribution density energy[ 1]. </summary>
///
/// <value>	The ff flux contribution density energy[ 1]. </value>
////////////////////////////////////////////////////////////////////////////////////////////////////

__constant__ double3 ff_flux_contribution_density_energy[1];

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Cuda initialize variables. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="nelr">			The nelr. </param>
/// <param name="variables">	[in,out] If non-null, the variables. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void cuda_initialize_variables(int nelr, double* variables)
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

void initialize_variables(int nelr, double* variables)
{
	dim3 Dg(nelr / block_length), Db(block_length);
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

__device__ __host__ inline void compute_flux_contribution(double& density, double3& momentum, double& density_energy, double& pressure, double3& velocity, double3& fc_momentum_x, double3& fc_momentum_y, double3& fc_momentum_z, double3& fc_density_energy)
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

	double de_p = density_energy+pressure;
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

__device__ inline void compute_velocity(double& density, double3& momentum, double3& velocity)
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

__device__ inline double compute_speed_sqd(double3& velocity)
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

__device__ inline double compute_pressure(double& density, double& density_energy, double& speed_sqd)
{
	return (double(GAMMA)-double(1.0))*(density_energy - double(0.5)*density*speed_sqd);
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

__device__ inline double compute_speed_of_sound(double& density, double& pressure)
{
	return sqrt(double(GAMMA)*pressure/density);
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

__global__ void cuda_compute_step_factor(int nelr, double* variables, double* areas, double* step_factors)
{
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);

	double density = variables[i + VAR_DENSITY*nelr];
	double3 momentum;
	momentum.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	momentum.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	momentum.z = variables[i + (VAR_MOMENTUM+2)*nelr];
	
	double density_energy = variables[i + VAR_DENSITY_ENERGY*nelr];
	
	double3 velocity;       compute_velocity(density, momentum, velocity);
	double speed_sqd      = compute_speed_sqd(velocity);
	double pressure       = compute_pressure(density, density_energy, speed_sqd);
	double speed_of_sound = compute_speed_of_sound(density, pressure);

	// dt = double(0.5) * sqrt(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
	step_factors[i] = double(0.5) / (sqrt(areas[i]) * (sqrt(speed_sqd) + speed_of_sound));
}

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

void compute_step_factor(int nelr, double* variables, double* areas, double* step_factors)
{
	dim3 Dg(nelr / block_length), Db(block_length);
    cudaEventRecord(start, 0);
	cuda_compute_step_factor<<<Dg, Db>>>(nelr, variables, areas, step_factors);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    kernelTime += elapsed * 1.e-3;
    CHECK_CUDA_ERROR();
}

/*

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

 *
 *
*/
__global__ void cuda_compute_flux(int nelr, int* elements_surrounding_elements, double* normals, double* variables, double* fluxes)
{
	const double smoothing_coefficient = double(0.2f);
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);
	
	int j, nb;
	double3 normal; double normal_len;
	double factor;
	
	double density_i = variables[i + VAR_DENSITY*nelr];
	double3 momentum_i;
	momentum_i.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	momentum_i.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	momentum_i.z = variables[i + (VAR_MOMENTUM+2)*nelr];

	double density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];

	double3 velocity_i;             				compute_velocity(density_i, momentum_i, velocity_i);
	double speed_sqd_i                          = compute_speed_sqd(velocity_i);
	double speed_i                              = sqrt(speed_sqd_i);
	double pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
	double speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
	double3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
	double3 flux_contribution_i_density_energy;	
	compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z, flux_contribution_i_density_energy);
	
	double flux_i_density = double(0.0);
	double3 flux_i_momentum;
	flux_i_momentum.x = double(0.0);
	flux_i_momentum.y = double(0.0);
	flux_i_momentum.z = double(0.0);
	double flux_i_density_energy = double(0.0);
		
	double3 velocity_nb;
	double density_nb, density_energy_nb;
	double3 momentum_nb;
	double3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
	double3 flux_contribution_nb_density_energy;	
	double speed_sqd_nb, speed_of_sound_nb, pressure_nb;
	
	#pragma unroll
	for(j = 0; j < NNB; j++)
	{
		nb = elements_surrounding_elements[i + j*nelr];
		normal.x = normals[i + (j + 0*NNB)*nelr];
		normal.y = normals[i + (j + 1*NNB)*nelr];
		normal.z = normals[i + (j + 2*NNB)*nelr];
		normal_len = sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
		
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
			factor = -normal_len*smoothing_coefficient*double(0.5)*(speed_i + sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
			flux_i_density += factor*(density_i-density_nb);
			flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
			flux_i_momentum.x += factor*(momentum_i.x-momentum_nb.x);
			flux_i_momentum.y += factor*(momentum_i.y-momentum_nb.y);
			flux_i_momentum.z += factor*(momentum_i.z-momentum_nb.z);

			// accumulate cell-centered fluxes
			factor = double(0.5)*normal.x;
			flux_i_density += factor*(momentum_nb.x+momentum_i.x);
			flux_i_density_energy += factor*(flux_contribution_nb_density_energy.x+flux_contribution_i_density_energy.x);
			flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.x+flux_contribution_i_momentum_x.x);
			flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.x+flux_contribution_i_momentum_y.x);
			flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.x+flux_contribution_i_momentum_z.x);
			
			factor = double(0.5)*normal.y;
			flux_i_density += factor*(momentum_nb.y+momentum_i.y);
			flux_i_density_energy += factor*(flux_contribution_nb_density_energy.y+flux_contribution_i_density_energy.y);
			flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.y+flux_contribution_i_momentum_x.y);
			flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.y+flux_contribution_i_momentum_y.y);
			flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.y+flux_contribution_i_momentum_z.y);
			
			factor = double(0.5)*normal.z;
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
			factor = double(0.5)*normal.x;
			flux_i_density += factor*(ff_variable[VAR_MOMENTUM+0]+momentum_i.x);
			flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].x+flux_contribution_i_density_energy.x);
			flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].x + flux_contribution_i_momentum_x.x);
			flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].x + flux_contribution_i_momentum_y.x);
			flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].x + flux_contribution_i_momentum_z.x);
			
			factor = double(0.5)*normal.y;
			flux_i_density += factor*(ff_variable[VAR_MOMENTUM+1]+momentum_i.y);
			flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].y+flux_contribution_i_density_energy.y);
			flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].y + flux_contribution_i_momentum_x.y);
			flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].y + flux_contribution_i_momentum_y.y);
			flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].y + flux_contribution_i_momentum_z.y);

			factor = double(0.5)*normal.z;
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

void compute_flux(int nelr, int* elements_surrounding_elements, double* normals, double* variables, double* fluxes)
{
	dim3 Dg(nelr / block_length), Db(block_length);
    cudaEventRecord(start, 0);
	cuda_compute_flux<<<Dg,Db>>>(nelr, elements_surrounding_elements, normals, variables, fluxes);
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

__global__ void cuda_time_step(int j, int nelr, double* old_variables, double* variables, double* step_factors, double* fluxes)
{
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);

	double factor = step_factors[i]/double(RK+1-j);

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

void time_step(int j, int nelr, double* old_variables, double* variables, double* step_factors, double* fluxes)
{
	dim3 Dg(nelr / block_length), Db(block_length);
    cudaEventRecord(start, 0);
	cuda_time_step<<<Dg,Db>>>(j, nelr, old_variables, variables, step_factors, fluxes);
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
    printf("Running CFDSolver (double)\n");
    bool quiet = op.getOptionBool("quiet");
    if(!quiet) {
        printf("WG size of %d\n", block_length);
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
		double h_ff_variable[NVAR];
		const double angle_of_attack = double(3.1415926535897931 / 180.0) * double(deg_angle_of_attack);
		
		h_ff_variable[VAR_DENSITY] = double(1.4);
		
		double ff_pressure = double(1.0);
		double ff_speed_of_sound = sqrt(GAMMA*ff_pressure / h_ff_variable[VAR_DENSITY]);
		double ff_speed = double(ff_mach)*ff_speed_of_sound;
		
		double3 ff_velocity;
		ff_velocity.x = ff_speed*double(cos((double)angle_of_attack));
		ff_velocity.y = ff_speed*double(sin((double)angle_of_attack));
		ff_velocity.z = 0.0;
		
		h_ff_variable[VAR_MOMENTUM+0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
		h_ff_variable[VAR_MOMENTUM+1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
		h_ff_variable[VAR_MOMENTUM+2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;
				
		h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY]*(double(0.5)*(ff_speed*ff_speed)) + (ff_pressure / double(GAMMA-1.0));

		double3 h_ff_momentum;
		h_ff_momentum.x = *(h_ff_variable+VAR_MOMENTUM+0);
		h_ff_momentum.y = *(h_ff_variable+VAR_MOMENTUM+1);
		h_ff_momentum.z = *(h_ff_variable+VAR_MOMENTUM+2);
		double3 h_ff_flux_contribution_momentum_x;
		double3 h_ff_flux_contribution_momentum_y;
		double3 h_ff_flux_contribution_momentum_z;
		double3 h_ff_flux_contribution_density_energy;
		compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum, h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity, h_ff_flux_contribution_momentum_x, h_ff_flux_contribution_momentum_y, h_ff_flux_contribution_momentum_z, h_ff_flux_contribution_density_energy);

		// copy far field conditions to the gpu
        cudaEventRecord(start, 0);

		CUDA_SAFE_CALL( cudaMemcpyToSymbol(ff_variable,          h_ff_variable,          NVAR*sizeof(double)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(ff_flux_contribution_momentum_x, &h_ff_flux_contribution_momentum_x, sizeof(double3)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(ff_flux_contribution_momentum_y, &h_ff_flux_contribution_momentum_y, sizeof(double3)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(ff_flux_contribution_momentum_z, &h_ff_flux_contribution_momentum_z, sizeof(double3)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(ff_flux_contribution_density_energy, &h_ff_flux_contribution_density_energy, sizeof(double3)) );		

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        transferTime += elapsed * 1.e-3;
	}
	int nel;
	int nelr;
	
	// read in domain geometry
	double* areas;
	int* elements_surrounding_elements;
	double* normals;
	{
        string inputFile = op.getOptionString("inputFile");
		std::ifstream file(inputFile.c_str());
        
        if(inputFile != "") {
		    file >> nel;
        } else {
            int problemSizes[4] = {97000, 200000, 1000000, 4000000};
            nel = problemSizes[op.getOptionInt("size") - 1];
        }
		nelr = block_length*((nel / block_length )+ std::min(1, nel % block_length));

		double* h_areas = new double[nelr];
		int* h_elements_surrounding_elements = new int[nelr*NNB];
		double* h_normals = new double[nelr*NDIM*NNB];

        srand(SEED);
				
		// read in data
		for(int i = 0; i < nel; i++)
		{
            if(inputFile != "") {
		    	file >> h_areas[i];
            } else {
                h_areas[i] = 1.0 * rand() / RAND_MAX;
            }
			for(int j = 0; j < NNB; j++)
			{
                if(inputFile != "") {
				    file >> h_elements_surrounding_elements[i + j*nelr];
                } else {
                    int val = i + (rand() % 20) - 10;
                    h_elements_surrounding_elements[i + j * nelr] = val;
                }
				if(h_elements_surrounding_elements[i+j*nelr] < 0) h_elements_surrounding_elements[i+j*nelr] = -1;
				h_elements_surrounding_elements[i + j*nelr]--; //it's coming in with Fortran numbering				
				
				for(int k = 0; k < NDIM; k++)
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
		
		areas = alloc<double>(nelr);
		upload<double>(areas, h_areas, nelr);

		elements_surrounding_elements = alloc<int>(nelr*NNB);
		upload<int>(elements_surrounding_elements, h_elements_surrounding_elements, nelr*NNB);

		normals = alloc<double>(nelr*NDIM*NNB);
		upload<double>(normals, h_normals, nelr*NDIM*NNB);
				
		delete[] h_areas;
		delete[] h_elements_surrounding_elements;
		delete[] h_normals;
	}

	// Create arrays and set initial conditions
	double* variables = alloc<double>(nelr*NVAR);
	initialize_variables(nelr, variables);

	double* old_variables = alloc<double>(nelr*NVAR);   	
	double* fluxes = alloc<double>(nelr*NVAR);
	double* step_factors = alloc<double>(nelr); 

	// make sure all memory is doublely allocated before we start timing
	initialize_variables(nelr, old_variables);
	initialize_variables(nelr, fluxes);
	cudaMemset( (void*) step_factors, 0, sizeof(double)*nelr );
	// make sure CUDA isn't still doing something before we start timing
	cudaDeviceSynchronize();

	// these need to be computed the first time in order to compute time step

	// Begin iterations
	for(int i = 0; i < iterations; i++)
	{
		copy<double>(old_variables, variables, nelr*NVAR);
		
		// for the first iteration we compute the time step
		compute_step_factor(nelr, variables, areas, step_factors);
        CHECK_CUDA_ERROR();
		
		for(int j = 0; j < RK; j++)
		  {
		    compute_flux(nelr, elements_surrounding_elements, normals, variables, fluxes);
            CHECK_CUDA_ERROR();
		    time_step(j, nelr, old_variables, variables, step_factors, fluxes);
            CHECK_CUDA_ERROR();
		  }
	}

	cudaDeviceSynchronize();

    if(op.getOptionBool("verbose")) {
	    dump(variables, nel, nelr);
    }

	
	dealloc<double>(areas);
	dealloc<int>(elements_surrounding_elements);
	dealloc<double>(normals);
	
	dealloc<double>(variables);
	dealloc<double>(old_variables);
	dealloc<double>(fluxes);
	dealloc<double>(step_factors);

    char atts[1024];
    sprintf(atts, "numelements:%d", nel);
    resultDB.AddResult("cfd_double_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult("cfd_double_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("cfd_double_parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime+transferTime);
}
