#include "Matrix.h"
#include "Timer.h"
#include <complex>

using namespace MatrixLibrary;
using namespace std;

int main()
{
	Vec4 v{1, 324, -12, 1};
	cout << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << endl;
	Mat4 t = Mat4::identity();
	t = t.translate(6, 3, -3);
	v = t * v;
	cout << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << endl;
}
	