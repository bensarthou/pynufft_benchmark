from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules=[
	Extension("thread_demo",
			  ["NDFT_2_c.pyx"],
			  libraries=["m"],
			  extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
			  extra_link_args=['-fopenmp'],
			  include_dirs=[numpy.get_include()]
			  )
]

setup(
  name = "NDFT with multiprocessing",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)
