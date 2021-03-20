#pragma once
#include <tuple>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <random>
#include <typeinfo>
#include <streambuf>
#include <regex>

// TODO: Add resize method? or at least figure out whether or not
// a feature is necessary. Assignment has the same functionality, 
// but a resize() method could be more intuitive.
// TODO: Check validity of matrix-matrix operations now that 
// m_rowMajor has been implemented. Using (*this)[i] could cause
// issues
namespace MatrixLibrary
{

	template<class T>
	class Matrix
	{
	  protected:
		using MathFunction = T (*)(T);
	    int m_rows;
		int m_columns;
		int m_size;
		T *m_data;
		const bool m_rowMajor;

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

	  public:

#pragma region Constructors / Assignment

		Matrix(int r, int c, bool rowMajor = true)
			:m_rows{ r }, m_columns{ c }, m_size{ r * c }, m_data{ new T[m_size] {} }, m_rowMajor{ rowMajor }
		{
			if (r < 0 || c < 0)
			{
				std::cerr << "Cannot construct a matrix of size " << shape() << std::endl;
				throw std::exception();
			}
		}

		Matrix(const Matrix& mat)
			:Matrix{ mat.m_rows, mat.m_columns, mat.m_rowMajor }
		{
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
				if (!mat.isVector() && m_rowMajor != mat.m_rowMajor)
				{
					std::cout << "Attempted to assign an rvalue " << (mat.m_rowMajor ? "row major" : "column major") 
						<< " matrix to a " << (m_rowMajor ? "row major" : "column major") << " matrix." << std::endl;

					throw std::exception();
				}

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

		static Matrix fromString(const std::string& str, const std::string& itemSplitter = ",", const std::string& rowSplitter = "\n", bool rowMajor = true)
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

		static Matrix fromFile(const std::string& fileName, const std::string& itemSplitter = ",", const std::string& rowSplitter = "\n", bool rowMajor = true)
		{
			std::ifstream i{ fileName };
			std::stringstream s;
			s << i.rdbuf();
			return fromString(s.str(), itemSplitter, rowSplitter, rowMajor);
		}

		static Matrix load(const std::string& fileName)
		{
			std::ifstream i{ fileName, std::ios::in | std::ios::binary };
			int rows = 0;
			int columns = 0;
			bool rowMajor = 0;
			i.read((char*)&rows, std::streamsize(sizeof(int)));
			i.read((char*)&columns, std::streamsize(sizeof(int)));
			i.read((char*)&rowMajor, std::streamsize(sizeof(bool)));

			Matrix result { rows, columns, rowMajor };

			i.read((char*)result.m_data, std::streamsize(result.m_size * sizeof(T)));

			return result;
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
			return m_columns == 1 || m_rows == 1;
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

		std::string toString(const std::string& itemDelimiter, const std::string& rowDelimiter, int precision = 3) const
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

		void toFile(const std::string& fileName, const std::string& itemDelimiter, const std::string& rowDelimiter, int precision = 3) const
		{
			std::ofstream f(fileName);
			f << toString(itemDelimiter, rowDelimiter);
		}

		void save(const std::string& fileName)
		{
			std::ofstream file { fileName, std::ios::out | std::ios::binary };
			file.write((char*)&m_rows, std::streamsize(sizeof(m_rows)));
			file.write((char*)&m_columns, std::streamsize(sizeof(m_columns)));
			file.write((char*)&m_rowMajor, std::streamsize(sizeof(m_rowMajor)));
			file.write((char*)m_data, std::streamsize(m_size * sizeof(T)));
			file.close();
		}

		T* data() const
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

		// TODO: Make this work with negative indexing?
		Matrix subMatrix(int rowStart, int rowEnd, int colStart, int colEnd) const
		{
			if (rowStart < 0 || rowStart > m_rows || colStart < 0 || colStart > m_columns ||
				rowEnd < 0 || rowEnd > m_rows || colEnd < 0 || colEnd > m_columns)
				throw std::exception();

			bool reverseRows = rowStart > rowEnd;
			bool reverseCols = colStart > colEnd;

			Matrix result{ abs(rowEnd - rowStart), abs(colEnd - colStart) };
			for (int i = rowStart; (reverseRows ? i > rowEnd : i < rowEnd); (reverseRows ? i-- : i++))
			{
				for (int j = colStart; (reverseCols ? j > colEnd : j < colEnd); (reverseCols ? j-- : j++))
				{
					result(abs(i - rowStart), abs(j - colStart)) = (*this)(i, j);
				}
			}
			return result;
		}

		Matrix subVector(int startIndex, int endIndex)
		{
			if (!isVector())
				throw std::exception();

			if (startIndex < 0 || startIndex > m_size || endIndex < 0 || endIndex > m_size)
				throw std::exception();

			bool reverse = startIndex > endIndex;

			int newSize = abs(endIndex - startIndex);
			Matrix result {(m_rows == 1 ? 1 : newSize), (m_columns == 1 ? 1 : newSize)};
			for (int i = startIndex; (reverse ? i > endIndex : i < endIndex); (reverse ? i-- : i++))
			{
				result[abs(i - startIndex)] = (*this)[i];
			}

			return result;
		}

		T max()
		{
			T max_val = (*this)[0];
			for (int i = 1; i < m_size; i++)
			{
				if ((*this)[i] > max_val)
				{
					max_val = (*this)[i];
				}
			}

			return max_val;
		}

		T min()
		{
			T min_val = (*this)[0];
			for (int i = 1; i < m_size; i++)
			{
				if ((*this)[i] < min_val)
				{
					min_val = (*this)[i];
				}
			}

			return min_val;
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
				throw std::exception();
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
				throw std::exception();
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
				throw std::exception();
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
				throw std::exception();
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
					throw std::exception();
				}

				m_rows = m_size / c;
				m_columns = c;
			}
			else if (c == -1)
			{
				if (m_size % r != 0)
				{
					std::cerr << "Cannot reshape matrix with shape " << shape() << " to have " << r << " rows" << std::endl;
					throw std::exception();
				}

				m_rows = r;
				m_columns = m_size / r;
			}
			else
			{
				if (m_size != r * c)
				{
					std::cerr << "The size of the matrix, " << m_size << "does not match the size given by r * c, " << r * c << std::endl;
					throw std::exception();
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
				throw std::exception();
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
			return const_cast<T&>(std::as_const(*this)(row, column));
		}

		const T& operator()(int row, int column) const
		{
			#ifndef NDEBUG
			// Check bounds only if debug
			if (row < -m_rows || row >= m_rows || column < -m_columns || column >= m_columns)
				throw std::exception();
			#endif
			
			if (m_rowMajor)
				return m_data[(row < 0 ? m_rows + row : row) * m_columns + (column < 0 ? m_columns + column : column)];
			else
				return m_data[(column < 0 ? m_columns + column : column) * m_rows + (row < 0 ? m_rows + row : row)];
		}

		T& operator[](int index)
		{
			return const_cast<T&>(std::as_const(*this)[index]);
		}

		const T& operator[](int index) const
		{
			#ifndef NDEBUG
			// Check bounds only if debug
			if (index < -m_size || index >= m_size)
				throw std::exception();
			#endif

			return m_data[(index < 0 ? m_size + index : index)];
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
				throw std::exception();

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
				throw std::exception();

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
				throw std::exception();
			}

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
				throw std::exception();

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