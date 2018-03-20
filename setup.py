from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'Non uniform Direct Fourier Transform',
  ext_modules = cythonize("NDFT_c.pyx"),
  include_dirs=[numpy.get_include()]
)
