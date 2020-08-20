#include "Matrix.h"
#include "Timer.h"
#include <complex>

using namespace MatrixLibrary;
using namespace std;

int main()
{
	Matrix<int> a{2, 2, true};
	Matrix<int> b{2, 2, true};

	a.randInit(0, 10);
	b.randInit(0, 10);

	(a * b).print();
}
	