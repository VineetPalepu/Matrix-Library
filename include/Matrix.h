#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 210
#include <tuple>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <random>
#include <typeinfo>
#include <streambuf>
#include <regex>
#include <CL/cl2.hpp>

#define DEVICE_GPU 0
#define DEVICE_CPU 1

// TODO: Get CL to not crash during long matrix multiplications
// figure out why CL crashes on large matrix multiplications
// Use multiple kernels to prevent long running kernel from
// stopping execution. Graphics drivers stop a process if 
// it hangs for a long time
// TODO: Add resize method? or at least figure out whether or not
// a feature is necessary. Assignment has the same functionality, 
// but a resize() method could be more intuitive.
namespace MatrixLibrary
{

	template<class T>
	class Matrix
	{
	  private:
		using MathFunction = T (*)(T);
	    int m_rows;
		int m_columns;
		int m_size;
		T *m_data;
		const bool m_rowMajor;

#pragma region OpenCL Static Initialization Code
		class OpenCL
		{
			friend class Matrix<T>;
		  private:
			static inline cl_context context;
			static inline cl_command_queue queue;
			static inline cl_device_id device;
			static inline const int DEVICE = DEVICE_CPU;
			static inline bool first = true;
			static inline std::string code;
			
			// TODO: Instead of using a DEVICE_GPU or DEVICE_CPU constant, change the clGetPlatformInfo() call
			// to query CL_DEVICE_TYPE_CPU or CL_DEVICE_TYPE_GPU instead
			static int clSetup()
			{
				if (!first)
					return 0;

				first = false;

				//std::cout << "Listing Platform Devices" << std::endl;
				cl_platform_id platform;
				clGetPlatformIDs(1, &platform, nullptr);
				char name[64];
				clGetPlatformInfo(platform, CL_PLATFORM_NAME, 64 * sizeof(char), name, NULL);
				//std::cout << "Platform:" << name << std::endl;
				cl_uint numDevices;
				clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
				if (numDevices == 0)
				{
					std::cout << "No valid OpenCL device found" << std::endl; 
				}
				cl_device_id *devices = new cl_device_id[numDevices];
				clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
				for (int i = 0; i < numDevices; i++)
				{
					clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 64 * sizeof(char), name, NULL);
					//std::cout << "\tDevice " << i << ": " << name << std::endl;
				}

				//std::cout << "Creating Context" << std::endl;
				cl_int err;
				context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &err);
				if (err != 0)
				{
					std::cout << "Error creating context: " << err << std::endl;
					delete[] devices;
					return -1;
				}

				device = devices[DEVICE];

				//std::cout << "Creating Command Queue" << std::endl;
				queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
				if (err != 0)
				{
					std::cout << "Error creating queue: " << err << std::endl;
					delete[] devices;
					return -1;
				}

				std::fstream inputFile;
				size_t length;
				inputFile.open("mat_mul_kernel.cl", std::ios::in | std::ios_base::binary);
				inputFile.seekg(0, std::ios::end);
				length = inputFile.tellg();
				inputFile.seekg(0, std::ios::beg);
				code = std::string((std::istreambuf_iterator<char>(inputFile)), std::istreambuf_iterator<char>());

				delete[] devices;
				return 0;
			}
		};
#pragma endregion

#pragma region Random
		template <class NumericType, typename Enable = void>
		class Random;

		template <class FloatingType>
		class Random<FloatingType, typename std::enable_if<std::is_floating_point<FloatingType>::value>::type>
		{
			std::random_device rand_dev;
			std::mt19937 generator;
			std::uniform_real_distribution<FloatingType> distr;

		public:
			Random(FloatingType min, FloatingType max)
				: rand_dev{}, generator{rand_dev()}, distr{min, max}
			{
			}

			FloatingType getNext()
			{
				return distr(generator);
			}
		};
		template <class IntegralType>
		class Random<IntegralType, typename std::enable_if<std::is_integral<IntegralType>::value>::type>
		{
			std::random_device rand_dev;
			std::mt19937 generator;
			std::uniform_int_distribution<IntegralType> distr;

		public:
			Random(IntegralType min, IntegralType max)
				: rand_dev{}, generator{rand_dev()}, distr{min, max}
			{
			}

			IntegralType getNext()
			{
				return distr(generator);
			}
		};
#pragma endregion

#pragma region Exceptions
		class INDEX_OUT_OF_BOUNDS{};
		class MATRIX_SHAPE_ERROR{};
		class NO_OPENCL_IMPLEMENTATION_FOUND{};
		class NO_OPENCL_DEVICE_FOUND{};
		class OPENCL_ERROR{};
#pragma endregion

#pragma region OpenCL Initializer
		struct Initializer
		{
			Initializer()
			{
				if (OpenCL::clSetup() == 0 && Matrix::createProgram() == 0)
					supportsCL = true;
			}
		};

		static inline Initializer init = Initializer();
		static inline cl_program program;
		static inline cl_kernel mat_mul_kernel;
		static inline bool supportsCL = false;

		static int createProgram()
		{
			if (typeid(T) == typeid(char) ||
				typeid(T) == typeid(unsigned char) ||
				typeid(T) == typeid(short) ||
				typeid(T) == typeid(unsigned short) ||
				typeid(T) == typeid(int) || 
				typeid(T) == typeid(unsigned int) ||
				typeid(T) == typeid(long) || 
				typeid(T) == typeid(unsigned long) || 
				typeid(T) == typeid(float) || 
				typeid(T) == typeid(double))
			{
				supportsCL = true;
			}
			else
			{
				supportsCL = false;
				return -1;
			}
			//std::cout << "Creating Program" << std::endl;
			cl_int err;

			OpenCL::code = std::regex_replace(OpenCL::code, std::regex("T"), typeid(T).name());
			const char* src = OpenCL::code.c_str();

			program = clCreateProgramWithSource(OpenCL::context, 1, &src, NULL, &err);
			if (err != 0)
			{
				std::cout << "Error creating program: " << err << std::endl;
				return -1;
			}

			//std::cout << "Building Program" << std::endl;
			err = clBuildProgram(program, 1, &OpenCL::device, NULL, NULL, NULL);
			if (err != 0)
			{
				std::cout << "Error building program: " << err << std::endl;
				if (err == -11)
				{
					size_t log_size;
					clGetProgramBuildInfo(program, OpenCL::device, CL_PROGRAM_BUILD_LOG, NULL, NULL, &log_size);
					char* log = new char[log_size];
					clGetProgramBuildInfo(program, OpenCL::device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
					std::cout << std::endl << log << std::endl;
					delete[] log;
				}
				return -1;
			}

			//std::cout << "Creating Kernel" << std::endl;
			mat_mul_kernel = clCreateKernel(program, "matMul", &err);
			if (err != 0)
			{
				std::cout << "Error creating kernel: " << err << std::endl;
				return -1;
			}
		}

		static Matrix clMul(const Matrix &mat1, const Matrix &mat2)
		{
			Matrix result{mat1.rows(), mat2.columns()};
			//cout << "Creating Buffers" << endl;
			cl_int err;
			cl_mem mat1Buf = clCreateBuffer(OpenCL::context, CL_MEM_READ_ONLY, sizeof(T) * mat1.size(), NULL, &err);
			if (err != 0)
			{
				std::cout << "Error creating mat1Buf: " << err << std::endl;
				throw OPENCL_ERROR();
			}
			cl_mem mat2Buf = clCreateBuffer(OpenCL::context, CL_MEM_READ_ONLY, sizeof(T) * mat2.size(), NULL, &err);
			if (err != 0)
			{
				std::cout << "Error creating mat2Buf: " << err << std::endl;
				throw OPENCL_ERROR();
			}
			cl_mem resultBuf = clCreateBuffer(OpenCL::context, CL_MEM_WRITE_ONLY, sizeof(T) * result.size(), NULL, &err);
			if (err != 0)
			{
				std::cout << "Error creating resultBuf: " << err << std::endl;
				throw OPENCL_ERROR();
			}

			//cout << "Copying Data to Buffers" << endl;
			err = clEnqueueWriteBuffer(OpenCL::queue, mat1Buf, CL_TRUE, 0, mat1.size() * sizeof(T), mat1.m_data, 0, NULL, NULL);
			if (err != 0)
			{
				std::cout << "Error writing to mat1Buf: " << err << std::endl;
				throw OPENCL_ERROR();
			}
			err = clEnqueueWriteBuffer(OpenCL::queue, mat2Buf, CL_TRUE, 0, mat2.size() * sizeof(T), mat2.m_data, 0, NULL, NULL);
			if (err != 0)
			{
				std::cout << "Error writing to mat2Buf: " << err << std::endl;
				throw OPENCL_ERROR();
			}
			int m1_rows = mat1.rows();
			int m1_columns = mat1.columns();
			int m2_columns = mat2.columns();
			//cout << "Setting Kernel Arguments" << endl;
			err = clSetKernelArg(mat_mul_kernel, 0, sizeof(mat1Buf), &mat1Buf);
			if (err != 0)
			{
				std::cout << "Error setting arg 0: " << err << std::endl;
				throw OPENCL_ERROR();
			}
			err = clSetKernelArg(mat_mul_kernel, 1, sizeof(mat2Buf), &mat2Buf);
			if (err != 0)
			{
				std::cout << "Error setting arg 1: " << err << std::endl;
				throw OPENCL_ERROR();
			}
			err = clSetKernelArg(mat_mul_kernel, 2, sizeof(int), &m1_rows);
			if (err != 0)
			{
				std::cout << "Error setting arg 2: " << err << std::endl;
				throw OPENCL_ERROR();
			}
			err = clSetKernelArg(mat_mul_kernel, 3, sizeof(int), &m1_columns);
			if (err != 0)
			{
				std::cout << "Error setting arg 3: " << err << std::endl;
				throw OPENCL_ERROR();
			}
			err = clSetKernelArg(mat_mul_kernel, 4, sizeof(int), &m2_columns);
			if (err != 0)
			{
				std::cout << "Error setting arg 4: " << err << std::endl;
				throw OPENCL_ERROR();
			}
			err = clSetKernelArg(mat_mul_kernel, 5, sizeof(resultBuf), &resultBuf);
			if (err != 0)
			{
				std::cout << "Error setting arg 5: " << err << std::endl;
				throw OPENCL_ERROR();
			}

			//cout << "Enqueueing Kernel" << endl;

			const int DIMS = 2;
			size_t *g_work_size = new size_t[DIMS];
			g_work_size[0] = ((mat1.rows() / 16) + 1) * 16;
			g_work_size[1] = ((mat2.columns() / 16) + 1) * 16;
			size_t *l_work_size = new size_t[DIMS];
			l_work_size[0] = 16;
			l_work_size[1] = 16;
			err = clEnqueueNDRangeKernel(OpenCL::queue, mat_mul_kernel, DIMS, NULL, g_work_size, l_work_size, 0, NULL, NULL);
			if (err != 0)
			{
				std::cout << "Error enqueueing kernel: " << err << std::endl;
				delete[] g_work_size;
				delete[] l_work_size;

				throw OPENCL_ERROR();
			}
			delete[] g_work_size;
			delete[] l_work_size;

			//cout << "Copying Buffer to Host" << endl;
			err = clEnqueueReadBuffer(OpenCL::queue, resultBuf, CL_TRUE, 0, sizeof(T) * result.size(), result.m_data, 0, NULL, NULL);
			if (err != 0)
			{
				std::cout << "Error reading resultBuf: " << err << std::endl;
				throw OPENCL_ERROR();
			}

			clReleaseMemObject(mat1Buf);
			clReleaseMemObject(mat2Buf);
			clReleaseMemObject(resultBuf);

			return result;
		}

		static void clFlushAll() // call at end of program (automatically?)
		{
			clFlush(OpenCL::queue);
			clFinish(OpenCL::queue);
			clReleaseKernel(mat_mul_kernel);
			clReleaseProgram(program);
			clReleaseCommandQueue(OpenCL::queue);
			clReleaseContext(OpenCL::context);
		}
#pragma endregion

	  public:

#pragma region Constructors / Assignment

		Matrix(int r, int c, bool rowMajor = true)
			:m_rows{ r }, m_columns{ c }, m_size{ r * c }, m_data{ new T[m_size] {} }, m_rowMajor{ rowMajor }
		{
			if (r < 0 || c < 0)
			{
				std::cerr << "Cannot construct a matrix of size " << shape() << std::endl;
				throw MATRIX_SHAPE_ERROR();
			}
		}

		Matrix(const Matrix& mat)
			:Matrix{ mat.m_rows, mat.m_columns, mat.m_rowMajor }
		{
			std::cout << "Copy Constructor" << std::endl;
			for (int i = 0; i < m_rows; i++)
			{
				for (int j = 0; j < m_columns; j++)
				{
					(*this)(i, j) = mat(i, j);
				}
			}
		}

		Matrix(Matrix&& mat)
			:m_rows{ mat.m_rows }, m_columns{ mat.m_columns }, m_size{ m_rows * m_columns }, m_data{ mat.m_data }, m_rowMajor{ mat.m_rowMajor}
		{
			mat.m_rows = 0;
			mat.m_columns = 0;
			mat.m_size = 0;
			mat.m_data = nullptr;
		}

		~Matrix()
		{
			delete[] m_data;
		}

		Matrix& operator=(const Matrix& mat)
		{
			if (this != &mat)
			{
				if (m_size != mat.m_size)
				{
					delete[] m_data;
					m_data = new T[mat.m_size];
					m_size = mat.m_size;
				}
				m_rows = mat.m_rows;
				m_columns = mat.m_columns;

				for (int i = 0; i < m_rows; i++)
				{
					for (int j = 0; j < m_columns; j++)
					{
						(*this)(i, j) = mat(i, j);
					}
				}
			}
			return *this;
		}


		Matrix& operator=(Matrix&& mat)
		{
			if (this != &mat)
			{
				delete[] m_data;

				m_rows = mat.m_rows;
				m_columns = mat.m_columns;
				m_size = mat.m_size;
				m_data = mat.m_data;
				m_rowMajor = mat.m_rowMajor;

				mat.m_data = nullptr;
			}
			return *this;
		}

		Matrix& operator=(T val)
		{
			for (int i = 0; i < m_rows; i++)
			{
				for(int j = 0; j < m_columns; j++)
				{
					(*this)(i, j) = val;
				}
			}
			return *this;
		}

		template <class U> operator Matrix<U>()
		{
			Matrix<U> result{ m_rows, m_columns, m_rowMajor };
			for (int i = 0; i < m_rows; i++)
			{
				for (int j = 0; j < m_columns; j++)
				{
					(*this)(i, j) = (U)(*this)(i, j);
				}
			}
			return result;
		}

		// Need to fix to make it work for column major matrices
		static Matrix fromString(const std::string& str, const std::string& itemSplitter, const std::string& rowSplitter, bool rowMajor)
		{
			int rows = count(str, rowSplitter) + 1;
			int columns = count(str.substr(0, str.find(rowSplitter)), itemSplitter) + 1;
			Matrix result{ rows, columns, rowMajor };

			int pos = 0;
			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < columns; j++)
				{
					size_t itemDelimitPos = str.find(itemSplitter, pos);
					size_t rowDelimitPos = str.find(rowSplitter, pos);

					int endPos = 0;
					int delimLength = 0;
					if (itemDelimitPos < rowDelimitPos)
					{
						endPos = itemDelimitPos;
						delimLength = itemSplitter.size();
					}
					else if (rowDelimitPos < itemDelimitPos)
					{
						endPos = rowDelimitPos;
						delimLength = rowSplitter.size();
					}
					else
					{
						endPos = itemDelimitPos;
						delimLength = std::max<size_t>(itemSplitter.size(), rowSplitter.size()); 
					}
					std::stringstream numstr{str.substr(pos, endPos)};
					pos = endPos + delimLength;
					
					T num = 0;
					numstr >> num;
					result(i, j) = num; 
				}
			}
			return result;
		}

		// make private once done
		static int count(const std::string& str, const std::string& substr)
		{
			int count = 0;
			int pos = -1;

			while ((pos = str.find(substr, pos + 1)) != std::string::npos)
			{
				pos += substr.size();
				count++;
			}

			return count;
		}

		static Matrix fromFile(const std::string& fileName, const std::string& itemSplitter, const std::string& rowSplitter, bool rowMajor)
		{
			std::ifstream i{ fileName };
			std::stringstream s;
			s << i.rdbuf();
			return fromString(s.str(), itemSplitter, rowSplitter, rowMajor);
		}

		static Matrix identity(int n)
		{
			Matrix result{ n, n };
			for (int i = 0; i < n; i++)
			{
				result(i, i) = 1;
			}
			return result;
		}

		Matrix& randInit(T min = 0, T max = 1)
		{
			Random<T> rand(min, max);

			for (int i = 0; i < m_rows; i++)
			{
				for (int j = 0; j < m_columns; j++)
				{
					(*this)(i, j) = rand.getNext();
				}
			}

			return *this;
		}
		
#pragma endregion

#pragma region Getters
		bool isVector() const
		{
			return m_columns == 1;
		}
		bool isSquare() const
		{
			return m_rows == m_columns;
		}
		int size() const
		{
			return m_size;
		}

		int rows() const
		{
			return m_rows;
		}

		int columns() const
		{
			return m_columns;
		}

		bool isRowMajor() const
		{
			return m_rowMajor;
		}

		friend std::ostream& operator<< (std::ostream& os, const Matrix<T>& mat)
		{
			os << std::setfill(' ');

			os << "[[";
			for (int i = 0; i < mat.m_rows; i++)
			{
				for (int j = 0; j < mat.m_columns; j++)
				{
					os << " " << std::setw(os.precision() + 3) << mat(i, j) << " ";
				}
				if (i != mat.m_rows - 1)
					os << "],\n [";
			}
			os << "]]\n";

			return os;
		}

		void print(int precision = 3) const
		{
			std::cout << std::setprecision(precision) << (*this);
		}

		void printArray(int precision = 3) const
		{
			std::cout << "[ ";
			for (int i = 0; i < m_size; i++)
			{
				std::cout << m_data[i] << " ";
			}
			std::cout << "]" << std::endl;
		}

		std::string shape() const
		{
			return "(" + std::to_string(m_rows) + ", " + std::to_string(m_columns) + ")";
		}

		std::string toString(std::string itemDelimiter, std::string rowDelimiter, int precision = 3) const
		{
			std::string s;
			s.reserve(m_size * 5);
			std::stringstream result(s);
			result.precision(precision);
			for (int i = 0; i < m_rows; i++)
			{
				for (int j = 0; j < m_columns; j++)
				{
					result << (*this)(i, j);
					if (j != m_columns - 1)
						result << itemDelimiter;
				}
				if (i != m_rows - 1)
					result << rowDelimiter;
			}

			return result.str();
		}

		void toFile(std::string fileName, std::string itemDelimiter, std::string rowDelimiter, int precision = 3) const
		{
			std::ofstream f(fileName);
			f << toString(itemDelimiter, rowDelimiter);
		}

		T* data()
		{
			return m_data;
		}

#pragma endregion

#pragma region Matrix Operations
		Matrix transpose() const
		{
			Matrix result{ m_columns, m_rows };
			for (int i = 0; i < result.m_rows; i++)
			{
				for (int j = 0; j < result.m_columns; j++)
				{
					result(i, j) = (*this)(j, i);
				}
			}
			return result;
		}

		Matrix subMatrix(int rowStart, int rowEnd, int colStart, int colEnd) const
		{
			if (rowStart > rowEnd || colStart > colEnd)
			{
				throw MATRIX_SHAPE_ERROR();
			}
			if (rowStart < 0 || colStart < 0 || rowEnd > m_rows || colEnd > m_columns)
			{
				throw INDEX_OUT_OF_BOUNDS();
			}

			Matrix result{ rowEnd - rowStart, colEnd - colStart };
			for (int i = rowStart; i < rowEnd; i++)
			{
				for (int j = colStart; j < colEnd; j++)
				{
					result(i - rowStart, j - colStart) = (*this)(i, j);
				}
			}
			return result;
		}

		Matrix row(int row) const
		{
			return subMatrix(row, row + 1, 0, m_columns);
		}

		Matrix column(int column) const
		{
			return subMatrix(0, m_rows, column, column + 1);
		}

		Matrix hStack(const Matrix& mat) const
		{
			if (m_rows != mat.m_rows)
			{
				std::cerr << "The number of rows does not match: " << shape() << ", " << mat.shape() << std::endl;
				throw MATRIX_SHAPE_ERROR();
			}

			Matrix result{ m_rows, m_columns + mat.m_columns };
			for (int i = 0; i < result.m_rows; i++)
			{
				for (int j = 0; j < result.m_columns; j++)
				{
					if (j < m_columns)
						result(i, j) = (*this)(i, j);
					else
						result(i, j) = mat(i, j - m_columns);
				}
			}

			return result;
		}

		std::tuple<Matrix, Matrix> hSplit(int splitColumn) const
		{
			if (splitColumn > m_columns)
			{
				std::cerr << "The supplied index, " << splitColumn << ", must be less than or equal to the number of columns, " << m_columns << std::endl;
				throw INDEX_OUT_OF_BOUNDS();
			}

			Matrix m1{ m_rows, splitColumn };
			Matrix m2{ m_rows, m_columns - splitColumn };

			for (int i = 0; i < m_rows; i++)
			{
				for (int j = 0; j < m_columns; j++)
				{
					if (j < splitColumn)
					{
						m1(i, j) = (*this)(i, j);
					}
					else
					{
						m2(i, j - splitColumn) = (*this)(i, j);
					}
				}
			}

			return { m1, m2 };
		}

		Matrix vStack(const Matrix& mat) const
		{
			if (m_columns != mat.m_columns)
			{
				std::cerr << "The number of columns does not match: " << shape() << ", " << mat.shape() << std::endl;
				throw MATRIX_SHAPE_ERROR();
			}

			Matrix result{ m_rows + mat.m_rows, m_columns };
			for (int i = 0; i < result.m_rows; i++)
			{
				for (int j = 0; j < result.m_columns; j++)
				{
					if (i < m_rows)
						result(i, j) = (*this)(i, j);
					else
						result(i, j) = mat(i - m_rows, j);
				}
			}

			return result;
		}

		std::tuple<Matrix, Matrix> vSplit(int splitRow) const
		{
			if (splitRow > m_columns)
			{
				std::cerr << "The supplied index, " << splitRow << ", must be less than or equal to the number of rows, " << m_rows << std::endl;
				throw INDEX_OUT_OF_BOUNDS();
			}

			Matrix m1{ splitRow, m_columns };
			Matrix m2{ m_rows - splitRow, m_columns };

			for (int i = 0; i < m_rows; i++)
			{
				for (int j = 0; j < m_columns; j++)
				{
					if (i < splitRow)
					{
						m1(i, j) = (*this)(i, j);
					}
					else
					{
						m2(i - splitRow, j) = (*this)(i, j);
					}
				}
			}

			return { m1, m2 };
		}

		Matrix reshape(int r, int c)
		{
			if (r == -1)
			{
				if (m_size % c != 0)
				{
					std::cerr << "Cannot reshape matrix with shape " << shape() << " to have " << c << " columns" << std::endl;
					throw MATRIX_SHAPE_ERROR();
				}

				m_rows = m_size / c;
				m_columns = c;
			}
			else if (c == -1)
			{
				if (m_size % r != 0)
				{
					std::cerr << "Cannot reshape matrix with shape " << shape() << " to have " << r << " rows" << std::endl;
					throw MATRIX_SHAPE_ERROR();
				}

				m_rows = r;
				m_columns = m_size / r;
			}
			else
			{
				if (m_size != r * c)
				{
					std::cerr << "The size of the matrix, " << m_size << "does not match the size given by r * c, " << r * c << std::endl;
					throw MATRIX_SHAPE_ERROR();
				}

				m_rows = r;
				m_columns = c;
			}

			return *this;
		}

		T sum() const
		{
			T sum = 0;

			for (int i = 0; i < m_rows; i++)
			{
				for (int j = 0; j < m_columns; j++)
				{
					sum += (*this)(i, j);
				}
			}

			return sum;
		}

		Matrix inverse() const
		{
			if (!isSquare())
			{
				std::cerr << "matrix of shape " << shape() << " is not square" << std::endl;
				throw MATRIX_SHAPE_ERROR();
			}

			Matrix augmented = hStack(identity(m_rows)).rowReduce();
			
			return augmented.subMatrix(0, m_columns, m_rows, m_columns * 2);
		}

		Matrix rowReduce() const
		{
			Matrix mat{ *this };
			for (int j = 0; j < mat.m_columns; j++)// start at first column
			{
				for (int i = j; i < mat.m_rows; i++) // start at jth row (lower diagonal half of matrix)
				{
					T val = mat(i, j);
					for (int k = 0; k < mat.m_columns; k++)
					{
						if (i == j)
							mat(i, k) = mat(i, k) / val;
						else
							mat(i, k) = mat(i, k) - val * mat(j, k);
					}
				}
				for (int i = 0; i < j; i++)
				{
					T val = mat(i, j);
					for (int k = 0; k < mat.m_columns; k++)
					{
						mat(i, k) = mat(i, k) - val * mat(j, k);
					}
				}
			}
			return mat;
		}
#pragma endregion

#pragma region Misc Operators
		T& operator()(int row, int column)
		{
			return const_cast<T&>(std::as_const(*this).operator()(row, column));
		}

		const T& operator()(int row, int column) const
		{
			if (m_rowMajor)
				return m_data[row * m_columns + column];
			else
				return m_data[column * m_rows + row];
		}

		T& operator[](int index)
		{
			return const_cast<T&>(std::as_const(*this).operator()[index]);
		}

		const T& operator[](int index) const
		{
			if (index < 0 || index >= m_size)
				throw INDEX_OUT_OF_BOUNDS();

			return m_data[index];
		}

		bool operator==(const Matrix& mat)
		{
			if (m_rows != mat.m_rows || m_columns != mat.m_columns)
				return false;

			for (int i = 0; i < m_rows; i++)
			{
				for (int j = 0; j < m_columns; j++)
				{
					if ((*this)(i, j) != mat(i, j))
						return false;
				}
			}
			return true;
		}

		bool operator!=(const Matrix& mat)
		{
			return !(*this == mat);
		}
#pragma endregion

#pragma region Math Operators
#pragma region Addition
		friend Matrix operator+(const Matrix& mat1, const Matrix& mat2)
		{
			if (mat1.m_rows != mat2.m_rows || mat1.m_columns != mat2.m_columns)
				throw MATRIX_SHAPE_ERROR();

			Matrix result{ mat1.m_rows,mat1.m_columns };
			for (int i = 0; i < result.m_rows; i++)
			{
				for (int j = 0; j < result.m_columns; j++)
				{
					result(i, j) = mat1(i, j) + mat2(i, j);
				}
			}

			return result;
		}

		friend Matrix operator+(const Matrix& mat, T scl)
		{
			Matrix result{ mat.m_rows, mat.m_columns };
			for (int i = 0; i < result.m_rows; i++)
			{
				for (int j = 0; j < result.m_columns; j++)
				{
					result(i, j) = mat(i, j) + scl;
				}
			}

			return result;
		}

		friend Matrix operator+(T scl, const Matrix& mat)
		{
			return mat + scl;
		}

		Matrix& operator+=(const Matrix& mat)
		{
			return *this = *this + mat;
		}

		Matrix& operator+=(T scl)
		{
			return *this = *this + scl;
		}

		Matrix operator+() const
		{
			return 0 + *this;
		}
#pragma endregion
#pragma region Subtraction
		friend Matrix operator-(const Matrix& mat1, const Matrix& mat2)
		{
			if (mat1.m_rows != mat2.m_rows || mat1.m_columns != mat2.m_columns)
				throw MATRIX_SHAPE_ERROR();

			Matrix result{ mat1.m_rows, mat1.m_columns };
			for (int i = 0; i < result.m_rows; i++)
			{
				for (int j = 0; j < result.m_columns; j++)
				{
					result(i, j) = mat1(i, j) - mat2(i, j);
				}
			}

			return result;
		}

		friend Matrix operator-(const Matrix& mat, T scl)
		{
			Matrix result{ mat.m_rows, mat.m_columns };
			for (int i = 0; i < result.m_rows; i++)
			{
				for (int j = 0; j < result.m_columns; j++)
				{
					result(i, j) = mat(i, j) - scl;
				}
			}

			return result;
		}

		friend Matrix operator-(T scl, const Matrix& mat)
		{
			Matrix result{ mat.m_rows, mat.m_columns };
			for (int i = 0; i < result.m_rows; i++)
			{
				for (int j = 0; j < result.m_columns; j++)
				{
					result(i, j) = scl - mat(i, j);
				}
			}

			return result;
		}

		Matrix& operator-=(const Matrix& mat)
		{
			return *this = *this - mat;
		}

		Matrix& operator-=(T scl)
		{
			return *this = *this - scl;
		}

		Matrix operator-() const
		{
			return 0 - *this;
		}
#pragma endregion
#pragma region Multiplication
		friend Matrix operator*(const Matrix& mat1, const Matrix& mat2)
		{
			if (mat1.m_columns != mat2.m_rows)
			{
				std::cerr << "attempted to multiply matrices of sizes " << mat1.shape() << ", " << mat2.shape() << std::endl;
				throw MATRIX_SHAPE_ERROR();
			}

			// TODO: implement clMul for matrices stored in column major order
			if (supportsCL && mat1.isRowMajor() && mat2.isRowMajor())
				return clMul(mat1, mat2);
			else
			{
				Matrix result{ mat1.m_rows, mat2.m_columns };
				for (int i = 0; i < result.m_rows; i++)
				{
					for (int j = 0; j < result.m_columns; j++)
					{
						T sum = 0;
						for (int k = 0; k < mat1.m_columns; k++)
						{
							sum += mat1(i, k) * mat2(k, j);
						}
						result(i, j) = sum;
					}
				}
				return result;
			}
		}

		friend Matrix operator*(const Matrix& mat, T scl)
		{
			Matrix result{ mat.m_rows, mat.m_columns };
			for (int i = 0; i < mat.m_rows; i++)
			{
				for (int j = 0; j < mat.m_columns; j++)
				{
					result(i, j) = mat(i, j) * scl;
				}
			}
			return result;
		}

		friend Matrix operator*(T scl, const Matrix& mat)
		{
			return mat * scl;
		}

		Matrix& operator*=(const Matrix& mat)
		{
			return *this = *this * mat;
		}

		Matrix& operator*=(T scl)
		{
			return *this = *this * scl;
		}

		Matrix scalarMultiply(const Matrix& mat) const
		{
			if (m_rows != mat.m_rows || m_columns != mat.m_columns)
				throw MATRIX_SHAPE_ERROR();

			Matrix result{ m_rows, m_columns };
			for (int i = 0; i < mat.m_rows; i++)
			{
				for (int j = 0; j < mat.m_columns; j++)
				{
					result(i, j) = (*this)(i, j) * mat(i, j);
				}
			}
			return result;
		}

#pragma endregion
#pragma region Division
		friend Matrix operator/(const Matrix& mat, T scl)
		{
			Matrix result{ mat.m_rows, mat.m_columns };
			for (int i = 0; i < mat.m_rows; i++)
			{
				for (int j = 0; j < mat.m_columns; j++)
				{
					result(i, j) = mat(i, j) / scl;
				}
			}
			return result;
		}

		friend Matrix operator/(T scl, const Matrix& mat)
		{
			Matrix result{ mat.m_rows, mat.m_columns };
			for (int i = 0; i < mat.m_rows; i++)
			{
				for (int j = 0; j < mat.m_columns; j++)
				{
					result(i, j) = scl / mat(i, j);
				}
			}
			return result;
		}

		Matrix& operator/=(T scl)
		{
			return *this = *this / scl;
		}
#pragma endregion
#pragma endregion
	};
}