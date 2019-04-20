#ifndef _SIMPLE_3D_ARRAY_H_
#define _SIMPLE_3D_ARRAY_H_

#include <cassert>

template <typename T>
class CSimple3DArray
{
public:
	CSimple3DArray(
        int x, 
        int y, 
        int z
        )
		:_d1(x), 
         _d2(y),          
         _d3(z), 
         _arraysize(x*y*z),
         _cells(new T[x*y*z])
	{
	}

	~CSimple3DArray()
	{
		delete [] _cells;
	}

	T& v(int i, int j, int k)
	{
		assert (i>=0 && i<_d1);
		assert (j>=0 && j<_d2);
		assert (k>=0 && k<_d3);
		int idx = i + j*_d1 + k*(_d1*_d2);
		return _cells[idx];
	}
	int dimension1() { return _d1; }
	int dimension2() { return _d2; }
	int dimension3() { return _d3; }
    int arraysize() { return _arraysize; }
	T *cells() { return _cells; }
private:
	int _d1; //Rows
	int _d2; //Cols
	int _d3; //Depth
    int _arraysize;
	T *_cells;
};

#endif
