# setup.py
import distutils
from distutils.core import setup, Extension

setup(name = "Fast Mandelbrot calculation",
      version = "1.0",
      ext_modules = [Extension("mandext", ["mandext.c"])])
