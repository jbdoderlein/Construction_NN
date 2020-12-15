from setuptools import setup, Extension
import os
import numpy

#os.environ["CC"] = "clang.exe"

ext = Extension(
    'c_extension',
    sources=['c_extension.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-lgomp'])

setup(name='c_extension',
      version='1.0',
      description='This is a demo package',
      ext_modules=[ext])
