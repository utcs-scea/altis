#include <cassert>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include "OptionParser.h"
#include "Utility.h"

using namespace std;

// Forward Declarations
void addBenchmarkSpecOptions(OptionParser &op);

int main(int argc, char *argv[])
{
    int ret = 0;
    try
    {
        // Get args
        OptionParser op;

        //Add shared options to the parser
	op.addOption("numRows", OPT_INT, "10000", "number of rows", 'r');
	op.addOption("numCols", OPT_INT, "10000", "number of columns", 'c');
	op.addOption("pattern", OPT_BOOL, "", "allow values to be randomly assigned", 'p');
	op.addOption("outfile", OPT_STRING, "input_Spmv", "specify output file", 'o');

	addBenchmarkSpecOptions(op);

	std::cout << "Hi lol" << std::endl;
	
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

