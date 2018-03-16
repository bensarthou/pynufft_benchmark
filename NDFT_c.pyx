import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t
# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function.
#
# The arrays f, g and h is typed as "np.ndarray" instances. The only effect
# this has is to a) insert checks that the function arguments really are
# NumPy arrays, and b) make some attribute access like f.shape[0] much
# more efficient. (In this example this doesn't matter though.)
import scipy.misc
import scipy.ndimage
import sys, time

def progress(count, total, status=''):
	cdef int bar_len = 60
	cdef int filled_len = int(round(bar_len * count / float(total)))
	cdef int percents = round(100.0 * count / float(total), 1)
	bar = '=' * filled_len + '-' * (bar_len - filled_len)
	sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
	sys.stdout.flush()

def ndft_1D(np.ndarray x, np.ndarray f, int N):
	"""non-equispaced discrete Fourier transform"""
	cdef int k = -(N // 2) + np.arange(N)
	return np.dot(f, np.exp(2j * np.pi * k * x[:, np.newaxis]))


def ndft_2D(np.ndarray x, np.ndarray f, int Nd):
	cdef int M = Nd[0]
	cdef int N = Nd[1]
	cdef int K = np.shape(x)[0]
	cdef np.ndarray ndft2d = np.array([0.0 for i in range(K)])
	for k in range(K):
		print('k',k ,'sur ', K)
		# progress(k, K)
		cdef double sum_ = 0.0
		for m in range(M):
			for n in range(N):
				# print(n,m)
				value = f[m, n]
				cdef double complex e = np.exp(- 1j * 2*np.pi * (x[k,0] + x[k,1]))
				sum_ += value * e
		ndft2d[k] = sum_ / M / N
	return ndft2d

def indft_2d(y, Nd, x):

	res = np.zeros(Nd)
	M,N = Nd[0], Nd[1]
	K = np.shape(x)[0]

	for m in range(M):
		for n in range(N):
			# print(n,m)
			sum_ = 0.0
			for k in range(K):
				e = np.exp(1j * 2*np.pi * (x[k,0] + x[k,1]))
				sum_ += y[k] * e
			pix = int(sum_.real + 0.5)
			res[m, n] = pix
	return res

def say_hello_to(name):
	print("Hello %s!" % name)
