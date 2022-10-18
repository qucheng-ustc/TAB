#python setup.py build_ext --inplace
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='eth2v4', ext_modules=cythonize("eth2v4.pyx"), include_dirs=[np.get_include()])
