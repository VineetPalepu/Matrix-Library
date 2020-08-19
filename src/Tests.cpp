#include "Matrix.h"
#include "Timer.h"
#include <complex>

using namespace MatrixLibrary;
using namespace std;

int main()
{
	Matrix<double> m{3, 2, true};
	m.randInit();
	m(0, 0) = 1.21345;
	m(0, 1) = 2;
	m(1, 0) = 3;
	m(1, 1) = 4;
	m(2, 0) = 5;
	m(2, 1) = 6;
	m.print();

	cout << endl;
	for (int i = 0; i < m.size(); i++)
		cout << m.begin()[i] << " ";
	cout << endl;
}