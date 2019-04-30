#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <set>
#include "fdtd3d.h"
#include "simple_3d_array.h"

using namespace std;

//FDTDGraphParams& graphParams,
bool testUnrolledGraph(FDTD_PARAMS params,
					   int numIterations, //Number of iterations of the field computation to perform
					   int numBlocks) //Number of distinct input blocks to push into the input
{
	CSimple3DArray<float> *Ex;
	CSimple3DArray<float> *Ey;
	CSimple3DArray<float> *Ez;

	CSimple3DArray<float> *Hx;
	CSimple3DArray<float> *Hy;
	CSimple3DArray<float> *Hz;

	CSimple3DArray<float> *ExOut;
	CSimple3DArray<float> *EyOut;
	CSimple3DArray<float> *EzOut;

	CSimple3DArray<float> *HxOut;
	CSimple3DArray<float> *HyOut;
	CSimple3DArray<float> *HzOut;

	std::map< std::string, CSimple3DArray<float>* > inputAddresses, outputAddresses;

	IterationSpaceSize HxSize = graphParams.findVarSize("Hx");
	configure_raw_array(HxSize.dim1, HxSize.dim2, HxSize.dim3, &Hx, -1);
	inputAddresses["Hx"] = Hx;
	configure_raw_array(HxSize.dim1, HxSize.dim2, HxSize.dim3, &HxOut, 0);
	outputAddresses["Hx"] = HxOut;

	IterationSpaceSize HySize = graphParams.findVarSize("Hy");
	configure_raw_array(HySize.dim1, HySize.dim2, HySize.dim3, &Hy, -1);
	inputAddresses["Hy"] = Hy;
	configure_raw_array(HySize.dim1, HySize.dim2, HySize.dim3, &HyOut, 0);
	outputAddresses["Hy"] = HyOut;

	IterationSpaceSize HzSize = graphParams.findVarSize("Hz");
	configure_raw_array(HzSize.dim1, HzSize.dim2, HzSize.dim3, &Hz, -1);
	inputAddresses["Hz"] = Hz;
	configure_raw_array(HzSize.dim1, HzSize.dim2, HzSize.dim3, &HzOut, 0);
	outputAddresses["Hz"] = HzOut;

	IterationSpaceSize ExSize = graphParams.findVarSize("Ex");
	configure_raw_array(ExSize.dim1, ExSize.dim2, ExSize.dim3, &Ex, -1);
	inputAddresses["Ex"] = Ex;
	configure_raw_array(ExSize.dim1, ExSize.dim2, ExSize.dim3, &ExOut, 0);
	outputAddresses["Ex"] = ExOut;

	IterationSpaceSize EySize = graphParams.findVarSize("Ey");
	configure_raw_array(EySize.dim1, EySize.dim2, EySize.dim3, &Ey, -1);
	inputAddresses["Ey"] = Ey;
	configure_raw_array(EySize.dim1, EySize.dim2, EySize.dim3, &EyOut, 0);
	outputAddresses["Ey"] = EyOut;

	IterationSpaceSize EzSize = graphParams.findVarSize("Ez");
	configure_raw_array(EzSize.dim1, EzSize.dim2, EzSize.dim3, &Ez, -1);
	inputAddresses["Ez"] = Ez;
	configure_raw_array(EzSize.dim1, EzSize.dim2, EzSize.dim3, &EzOut, 0);
	outputAddresses["Ez"] = EzOut;

	constructUnrolledGraph(graphParams, numIterations);
	for(size_t i=0; i<graphParams.paramInputs.size() ; ++i)
		graphParams.paramInputs.at(i)->SetNoDraw();
	graphParams.g->WriteDOTFile("graphfdtd.dot", false);
	graphParams.g->Run(g_bSingleThreaded);

	// Construct the required data blocks and push them into their channels
	CHighResolutionTimer * pTimer = new CHighResolutionTimer(gran_msec);
	pTimer->reset();
	for(int inputNum=0; inputNum<numBlocks ; ++inputNum)
	{
		for(std::map<std::string, std::vector< GraphInputChannel* > >::iterator iter  = graphParams.inputChannels.begin() ;
			iter != graphParams.inputChannels.end(); ++iter)
		{
			const std::string& varName = iter->first;
			const std::vector< GraphInputChannel* >& channels = iter->second;
			assert (inputAddresses.find(varName) != inputAddresses.end());
			Datablock *dataBlock = PTask::Runtime::AllocateDatablock(graphParams.findTemplate(varName),
                                                                     inputAddresses[varName]->cells(),
                                                                     inputAddresses[varName]->arraysize(),
                                                                     channels.at(0));
			for(size_t i=0; i<channels.size() ; ++i)
			{
				channels.at(i)->Push(dataBlock);
			}
			// We hold a reference implicitly to any blocks returned from AllocateDatablock
			// We have transferred ownership to the input channels we need to
			// release our references.
			dataBlock->Release();
		}
	}

	Datablock * constPrm = PTask::Runtime::AllocateDatablock(graphParams.pConstParamTemplate, &params, sizeof(params), NULL); //graphParams.paramInputs.at(0));
	for(size_t i=0; i<graphParams.paramInputs.size() ; ++i)
	{
		graphParams.paramInputs.at(i)->Push(constPrm);
	}
    constPrm->Release();
	cout << "Finished Pushing inputs\n";
	//Get the outputs of the graph
	for(int outputNum=0; outputNum<numBlocks ; ++outputNum)
	{
		for(std::map<std::string, GraphOutputChannel*>::iterator iter=graphParams.outputChannels.begin();
			iter!=graphParams.outputChannels.end(); ++iter)
		{
			const std::string& varName = iter->first;
			// std::cout << varName << std::endl;
			GraphOutputChannel *outputChannel = iter->second;
			Datablock * pResultBlock = outputChannel->Pull();
			assert (outputAddresses.find(varName) != outputAddresses.end());
			CSimple3DArray<float> *outputArray = outputAddresses[varName];
			pResultBlock->Lock();
			float* psrc = (float*) pResultBlock->GetDataPointer(FALSE);
			float* pdst = outputArray->cells();
			IterationSpaceSize varSize = graphParams.findVarSize(varName);
			int size = sizeof(float) * varSize.dim1 * varSize.dim2 * varSize.dim3;
			memcpy(pdst, psrc, size);
			pResultBlock->Unlock();
			pResultBlock->Release();
		}
	}
    double dGPUTime = pTimer->elapsed(false);

    double dCPUStart = pTimer->elapsed(false);
	fdtd_cpu_multiple_inputs(Hx, Hy, Hz, Ex, Ey, Ez, numIterations, numBlocks, params);
    double dCPUEnd = pTimer->elapsed(false);

	int numErrors = 20;
	// print_array(HxOut, "GPU:");
	// print_array(Hx,    "CPU:");
    bool bSuccess = true;
    double dTotalSqError = 0.0;
	printf("Hx: "); bSuccess &= check_array_result(Hx, HxOut, &numErrors, &dTotalSqError);
	printf("Hy: "); bSuccess &= check_array_result(Hy, HyOut, &numErrors, &dTotalSqError);
	printf("Hz: "); bSuccess &= check_array_result(Hz, HzOut, &numErrors, &dTotalSqError);
	printf("Ex: "); bSuccess &= check_array_result(Ex, ExOut, &numErrors, &dTotalSqError);
	printf("Ey: "); bSuccess &= check_array_result(Ey, EyOut, &numErrors, &dTotalSqError);
	printf("Ez: "); bSuccess &= check_array_result(Ez, EzOut, &numErrors, &dTotalSqError);
    if(!bSuccess && dTotalSqError > 1.0) {
        printf("failed\n");
    } else {
        printf("succeeded\n");
    }
    printf("GPU exec:\t%.1f\nCPU exec:\t%.1f\n", dGPUTime, dCPUEnd-dCPUStart);

    delete pTimer;
	delete Ex;
	delete Ey;
	delete Ez;
	delete Hx;
	delete Hy;
	delete Hz;

	delete ExOut;
	delete EyOut;
	delete EzOut;
	delete HxOut;
	delete HyOut;
	delete HzOut;
	return false;
}



int run_graph_fdtd_task(
	char * szdir,
	char * szshader,
	int Nx,
	int Ny,
	int Nz,
	int numBlocks,
	int iterations,
	KernelType ktype)
{
	//Nx = 32;
    //Ny = 32;
	//Nz = 4;

    //CONFIGUREPTASKU(UseCUDA, (ktype == CUDA));
    //CheckPlatformSupport((ktype == CUDA ? "quack.ptx":"quack.hlsl"), NULL);

    /*
    std::set<Datablock*> myset;
    std::set<Datablock*>::iterator vi;
    for(vi=myset.begin(); vi!=myset.end(); vi++) {
        printf("QUACK\n");
    }
    */

	//Hard coding the lengths of the cavities for now
	float Lx = float(0.05), Ly = float(0.04), Lz = float(0.03);
	float nrm = sqrt((Nx/Lx*Nx/Lx) + (Ny/Ly*Ny/Ly) + (Nz/Lz*Nz/Lz))*100;
	float pi = float(3.1415);

    printf("%d %d %d (%d iterations)\n",
           Nx,
           Ny,
           Nz,
           iterations);

    //FDTDGraphParams graphParams(Nx, Ny, Nz, std::string(szdir), ktype);
	FDTD_PARAMS params;

	params.Nx = Nx;
	params.Ny = Ny;
	params.Nz = Nz;

	params.Cx = Nx/Lx;
	params.Cy = Ny/Ly;
	params.Cz = Nz/Lz;

	params.eps0 = float(8.8541878e-12); // Permittivity of vacuum.
	params.mu0 = float(4e-7)*pi; //Permeability of vacuum.
	params.c0 = float(299792458); //Speed of light in vacuum.
	params.Dt = float(1.0)/(params.c0*nrm); //Time step.

	//graphParams.g = new Graph();

	//testSingleKernel("Ez", graphParams, params);
    /*
    if(g_bSingleThreaded || g_nChannelCapacity < numBlocks) {
        testUnrolledGraph_ST(graphParams, params, iterations, numBlocks);
    }
    */
    //else {
        //testUnrolledGraph(graphParams, params, iterations, numBlocks);
        testUnrolledGraph(params, iterations, numBlocks);
    //}

	//Do all the graph destruction stuff
    /*
	graphParams.g->Stop();
	graphParams.g->Teardown();
	Graph::DestroyGraph(graphParams.g);
	PTask::Runtime::Terminate();
    */
	return 0;
}
