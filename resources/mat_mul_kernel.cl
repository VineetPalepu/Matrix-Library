kernel void matMul(global T* mat1, global T* mat2, int mat1_r, int mat1_c, int mat2_c, global T* result)
{ 
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (i >= mat1_r || j >= mat2_c)
		return;

	T sum = 0;
	for (int k = 0; k < mat1_c; k++)
	{
		sum += mat1[i * mat1_c + k] * mat2[k * mat2_c + j];
	}
	result[i * mat2_c + j] = sum;
	
}