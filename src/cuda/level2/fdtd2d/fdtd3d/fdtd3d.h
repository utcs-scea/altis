#ifndef _FDTD3D_H_
#define _FDTD3D_H_

enum KernelType { HLSL, CUDA };

typedef struct _fdtdcb_t {
	int Nx;
	int Ny;
	int Nz;
	
	float Cx;
	float Cy;
	float Cz;

	float eps0; // = 8.8541878e-12; % Permittivity of vacuum.
	float mu0;  // = 4e-7*pi % Permeability of vacuum.
	float c0;   // = 299792458; % Speed of light in vacuum.
	float Dt;   // = 1/(c0*nrm); % Time step.
} FDTD_PARAMS;

struct IterationSpaceSize
{
	int dim1;
	int dim2;
	int dim3;
	IterationSpaceSize(int x=-1, int y=-1, int z=-1) :dim1(x), dim2(y), dim3(z) {}
};

int run_graph_fdtd_task(	
	char * szfile,
	char * szshader,
	int Nx,
	int Ny,
	int Nz,
	int numBlocks,
	int iterations,
	KernelType ktype
	);

#endif
