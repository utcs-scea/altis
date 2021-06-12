////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\common\main.cpp
//
// summary:	Implements the main class
// 
// origin: SHOC (https://github.com/vetter/shoc)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cstdlib>
#include <fstream>

#include <cuda.h> 
#include <cuda_runtime.h>

#include "ResultDatabase.h"
#include "OptionParser.h"
#include "Utility.h"
#include "cudacommon.h"

using namespace std;

// Forward Declarations
void addBenchmarkSpecOptions(OptionParser &op);
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op);

// ****************************************************************************
// Function: EnumerateDevicesAndChoose
//
// Purpose:
//   This function queries cuda about the available gpus in the system, prints
//   those results to standard out, and selects a device for use in the
//   benchmark.
//
// Arguments:
//   chooseDevice: logical number for the desired device
//   properties: whether or not to print device properties and exit
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation:
//
// Modifications:
//   Jeremy Meredith, Tue Oct  9 17:27:04 EDT 2012
//   Added a windows-specific --noprompt, which unless the user passes it,
//   prompts the user to press enter before the program exits on Windows.
//   This is because on Windows, the console disappears when the program
//   exits, but our results go to the console.
//
// ****************************************************************************
void EnumerateDevicesAndChoose(int chooseDevice, bool properties, bool quiet)
{
    cudaSetDevice(chooseDevice);
    int actualdevice;
    cudaGetDevice(&actualdevice);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (properties)
    {
        cout << "Number of devices = " << deviceCount << "\n";
    }
    string deviceName = "";
    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        if (device == actualdevice)
            deviceName = deviceProp.name;
        if (properties)
        {
            cout << "Device " << device << ":\n";
            cout << "  name               = '" << deviceProp.name << "'"
                    << endl;
            cout << "  totalGlobalMem     = " << HumanReadable(
                    deviceProp.totalGlobalMem) << endl;
            cout << "  sharedMemPerBlock  = " << HumanReadable(
                    deviceProp.sharedMemPerBlock) << endl;
            cout << "  regsPerBlock       = " << deviceProp.regsPerBlock
                    << endl;
            cout << "  warpSize           = " << deviceProp.warpSize << endl;
            cout << "  memPitch           = " << HumanReadable(
                    deviceProp.memPitch) << endl;
            cout << "  maxThreadsPerBlock = " << deviceProp.maxThreadsPerBlock
                    << endl;
            cout << "  maxThreadsDim[3]   = " << deviceProp.maxThreadsDim[0]
                    << "," << deviceProp.maxThreadsDim[1] << ","
                    << deviceProp.maxThreadsDim[2] << endl;
            cout << "  maxGridSize[3]     = " << deviceProp.maxGridSize[0]
                    << "," << deviceProp.maxGridSize[1] << ","
                    << deviceProp.maxGridSize[2] << endl;
            cout << "  totalConstMem      = " << HumanReadable(
                    deviceProp.totalConstMem) << endl;
            cout << "  major (hw version) = " << deviceProp.major << endl;
            cout << "  minor (hw version) = " << deviceProp.minor << endl;
            cout << "  clockRate          = " << deviceProp.clockRate << endl;
            cout << "  textureAlignment   = " << deviceProp.textureAlignment
                    << endl;
        }
    }
    if(properties) {
        return;
    }
    if(!quiet) {
        cout << "Chose device:"
            << " name='"<<deviceName<<"'"
            << " index="<<actualdevice
            << endl;
    }
}

void checkCudaFeatureAvailability(OptionParser &op) {
    int device = 0;
    checkCudaErrors(cudaGetDevice(&device));
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device));
    
    // Check UVM availability
    if (op.getOptionBool("uvm") || op.getOptionBool("uvm-advise") ||
            op.getOptionBool("uvm-prefetch") || op.getOptionBool("uvm-prefetch-advise")) {
        if (!deviceProp.unifiedAddressing) {
            std::cerr << "device doesn't support unified addressing, exiting..." << std::endl;
            safe_exit(-1);
        }
    }

    // Check Cooperative Group availability
    if (op.getOptionBool("coop")) {
        if (!deviceProp.cooperativeLaunch) {
            std::cerr << "device doesn't support cooperative kernels, exiting..." << std::endl;
            safe_exit(-1);
        }
    }

    // Check Dynamic Parallelism availability
    if (op.getOptionBool("dyn")) {
        int runtimeVersion = 0;
        checkCudaErrors(cudaRuntimeGetVersion(&runtimeVersion));
        if (runtimeVersion < 5000) {
            std::cerr << "CUDA runtime version less than 5.0, doesn't support \
                dynamic parallelism, exiting..." << std::endl;
            safe_exit(-1);
        }
    }

    // Check CUDA Graphs availability
    if (op.getOptionBool("graph")) {
        int runtimeVersion = 0;
        checkCudaErrors(cudaRuntimeGetVersion(&runtimeVersion));
        if (runtimeVersion < 10000) {
            std::cerr << "CUDA runtime version less than 10.0, doesn't support \
                CUDA Graph, exiting..." << std::endl;
            safe_exit(-1);
        }
    }
}

// ****************************************************************************
// Function: main
//
// Purpose:
//   The main function takes care of initialization (device and MPI),  then
//   performs the benchmark and prints results.
//
// Arguments:
//
//
// Programmer: Jeremy Meredith
// Creation:
//
// Modifications:
//   Jeremy Meredith, Wed Nov 10 14:20:47 EST 2010
//   Split timing reports into detailed and summary.  For serial code, we
//   report all trial values, and for parallel, skip the per-process vals.
//   Also detect and print outliers from parallel runs.
//
// ****************************************************************************
int main(int argc, char *argv[])
{
    int ret = 0;

    try
    {
        // Get args
        OptionParser op;

        // Add shared options to the parser
        op.addOption("properties", OPT_BOOL, "0",
                "show properties for available platforms and devices", 'p');
        op.addOption("device", OPT_VECINT, "0",
                "specify device(s) to run on", 'd');
        op.addOption("passes", OPT_INT, "10", "specify number of passes", 'n');
        op.addOption("size", OPT_INT, "1", "specify problem size", 's');
        op.addOption("verbose", OPT_BOOL, "0", "enable verbose output", 'v');
        op.addOption("quiet", OPT_BOOL, "0", "enable concise output", 'q');
        op.addOption("configFile", OPT_STRING, "", "path of configuration file", 'c');
        op.addOption("inputFile", OPT_STRING, "", "path of input file", 'i');
        op.addOption("outputFile", OPT_STRING, "", "path of output file", 'o');
        op.addOption("metricsFile", OPT_STRING, "", "path of file to write metrics to", 'm');

        // Add options for turn on/off CUDA features
        op.addOption("uvm", OPT_BOOL, "0", "enable CUDA Unified Virtual Memory, only demand paging");
        op.addOption("uvm-advise", OPT_BOOL, "0", "guide the driver about memory usage patterns");
        op.addOption("uvm-prefetch", OPT_BOOL, "0", "prefetch memory the specified destination device");
        op.addOption("uvm-prefetch-advise", OPT_BOOL, "0", "prefetch memory the specified destination device with memory guidance on");
        op.addOption("coop", OPT_BOOL, "0", "enable CUDA Cooperative Groups");
        op.addOption("dyn", OPT_BOOL, "0", "enable CUDA Dynamic Parallelism");
        op.addOption("graph", OPT_BOOL, "0", "enable CUDA Graphs");

        addBenchmarkSpecOptions(op);

        if (!op.parse(argc, argv))
        {
            op.usage();
            return (op.HelpRequested() ? 0 : 1);
        }

        bool properties = op.getOptionBool("properties");
        bool quiet = op.getOptionBool("quiet");
        string metricsfile = op.getOptionString("metricsFile");

        int device;
        device = op.getOptionVecInt("device")[0];
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (device >= deviceCount) {
            cerr << "Warning: device index: " << device <<
            " out of range, defaulting to device 0.\n";
            device = 0;
        }

        // Initialization
        EnumerateDevicesAndChoose(device, properties, quiet);
        if (properties)
        {
            return 0;
        }

        // Check CUDA feature availability
        checkCudaFeatureAvailability(op);

        ResultDatabase resultDB;

        // Run the benchmark
        RunBenchmark(resultDB, op);

        // If quiet, output overall result
        // else output metrics
        if (quiet) {
            resultDB.DumpOverall();
        } else {
            if (metricsfile.empty()) {
                cout << endl;
                resultDB.DumpSummary(cout);
            } else {
                ofstream ofs;
                ofs.open(metricsfile.c_str());
                resultDB.DumpCsv(metricsfile);
                ofs.close();
            }
        }
    }
    catch( std::exception& e )
    {
        std::cerr << e.what() << std::endl;
        ret = 1;
    }
    catch( ... )
    {
        ret = 1;
    }

    return ret;
}
