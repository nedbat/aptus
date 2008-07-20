#!/usr/bin/env python
"""\
Aptus: A Mandelbrot set explorer and renderer.

Aptus is a Mandelbrot set explorer and renderer with a wxPython GUI and
a computation extension in C for speed.
"""

import distutils
from distutils.core import setup, Extension

try:
    import numpy
except:
    raise Exception("Need numpy, from http://numpy.scipy.org/")

version = "2.0"

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
            sources=["ext/engine.c"],
            include_dirs=[numpy.get_include()],
            ),
        ],
    
    scripts = [
        'scripts/aptuscmd.py',
        'scripts/aptusgui.py',
        ],    
    )
