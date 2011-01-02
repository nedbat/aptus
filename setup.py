#!/usr/bin/env python
"""Aptus: A Mandelbrot set explorer and renderer.

Aptus is a Mandelbrot set explorer and renderer with a wxPython GUI and
a computation extension in C for speed.
"""

import distutils, sys
from distutils.core import setup, Extension
from distutils.cygwinccompiler import Mingw32CCompiler

try:
    import numpy
except:
    raise Exception("Need numpy, from http://numpy.scipy.org/")

version = "2.1"

doclines = __doc__.split("\n")

classifiers = """
Development Status :: 5 - Production/Stable
Environment :: Console
Environment :: MacOS X
Environment :: Win32 (MS Windows)
Environment :: X11 Applications :: GTK
License :: OSI Approved :: MIT License
Programming Language :: C
Programming Language :: Python
Topic :: Artistic Software
Topic :: Scientific/Engineering :: Mathematics
"""

# Most examples on the web seem to imply that O3 will be automatic,
# but for me it wasn't, and I want all the speed I can get...
extra_compile_args = ['-O3', '-fno-strict-aliasing']

if sys.platform == "win32":
    #if isinstance(self.compiler, Mingw32CCompiler):
        extra_compile_args = ['-O2']

setup(
    # The metadata
    name = "Aptus",
    description = doclines[0],
    long_description = "\n".join(doclines[2:]),
    version = version,
    author = "Ned Batchelder",
    author_email = "ned@nedbatchelder.com",
    url = "http://nedbatchelder.com/code/aptus",
    license = "MIT",
    classifiers = filter(None, classifiers.split("\n")),
    
    # The data
    packages = [
        'aptus',
        'aptus.gui',
        ],

    package_dir = {
        'aptus': 'src'
        },
    
    package_data = {
        'aptus': [
            '*.ico',
            '*.png',
            'palettes/*.ggr',
            ]
        },
    
    ext_modules = [
        Extension(
            "aptus.engine",
            sources=["ext/engine.cpp"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=extra_compile_args,
            ),
        ],
    
    scripts = [
        'scripts/aptuscmd.py',
        'scripts/aptusgui.py',
        ],
    )
