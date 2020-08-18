#include "Matrix.h"
#include "Timer.h"
#include <complex>

using namespace MatrixLibrary;
using namespace std;

int main()
{
	Matrix<double> m{4, 4};
	m.randInit();
	m(3, 3) = 0;
	m.print();
}