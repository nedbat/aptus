#!/usr/bin/env python
"""Aptus: A Mandelbrot set explorer and renderer.

Aptus is a Mandelbrot set explorer and renderer with a wxPython GUI and
a computation extension in C for speed.
"""

import distutils, sys
from distutils.core import Extension
from distutils.cygwinccompiler import Mingw32CCompiler
import glob

from setuptools import setup

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

data_files = []
options = {}

# Most examples on the web seem to imply that O3 will be automatic,
# but for me it wasn't, and I want all the speed I can get...
extra_compile_args = ['-O3']

if sys.platform == "win32":
    #if isinstance(self.compiler, Mingw32CCompiler):
    extra_compile_args = ['-O2']

    try:
        import py2exe
    except:
        py2exe = None

    if py2exe:
        data_files += [
            ("Microsoft.VC90.CRT", glob.glob(r'C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\redist\x86\Microsoft.VC90.CRT\*.*')),
            ]
        options.update({
            'py2exe': {
                'bundle_files': 1,
                },
            })

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
    classifiers = list(filter(None, classifiers.split("\n"))),

    # The data
    packages = [
        'aptus',
        'aptus.gui',
        "aptus.web",
        ],

    package_dir = {
        '': 'src'
        },

    package_data = {
        'aptus': [
            '*.ico',
            '*.png',
            'palettes/*.ggr',
            "web/static/*.*",
            ]
        },

    ext_modules = [
        Extension(
            "aptus.engine",
            sources=["ext/engine.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=extra_compile_args,
            ),
        ],

    entry_points = {
        "console_scripts": [
            "aptus = aptus.cmdline:main",
            "aptusgui = aptus.gui:main",
            "aptusweb = aptus.web:main",
            ],
        },

    install_requires = [
        "Pillow>=3.0",
        "numpy>=1.5.1",
        ],
    extras_require = {
        "gui": [
            "wxPython",
            ],
        "web": [
            "aiofiles",
            "fastapi",
            "jinja2",
            "uvicorn",
            "cachetools",
            ],
        },

    data_files = data_files,
    options = options,
    )
