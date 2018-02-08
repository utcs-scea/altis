#include <cassert>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include "OptionParser.h"

int main(int argc, char *argv[])
{
    int ret = 0;
    try
    {
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

