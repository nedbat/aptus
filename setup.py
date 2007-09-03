# setup.py
import distutils
from distutils.core import setup, Extension

version = "1.0"

setup(
    name = "Fast Mandelbrot calculation",
    version = version,
    ext_modules = [Extension("mandext", ["mandext.c"])]
    )
