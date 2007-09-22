# setup.py
import distutils
from distutils.core import setup, Extension
import numpy

version = "1.0"

setup(
    name = "altus",
    description = "Fast Mandelbrot calculation",
    version = version,
    ext_modules = [Extension(
        "mandext",
        sources=["mandext.c"],
        include_dirs=[numpy.get_include()],
        #extra_compile_args=['-O3', '-ffast-math'],
        )]
    )
