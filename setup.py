#!/usr/bin/env python
"""Aptus: A Mandelbrot set explorer and renderer.

Aptus is a Mandelbrot set explorer and renderer with a wxPython GUI and
a computation extension in C for speed.
"""

from setuptools import setup

import distutils
from distutils.core import Extension
import glob
import sys

try:
    import numpy
except:
    raise Exception("Need numpy, from http://numpy.scipy.org/")

version = "3.0"

doclines = __doc__.split("\n")

classifiers = """
Development Status :: 5 - Production/Stable
Environment :: Console
Environment :: MacOS X
Environment :: Win32 (MS Windows)
Environment :: X11 Applications :: GTK
Programming Language :: Python :: 3.9
License :: OSI Approved :: MIT License
Programming Language :: C
Programming Language :: Python
Topic :: Artistic Software
Topic :: Scientific/Engineering :: Mathematics
"""

options = {}

# Most examples on the web seem to imply that O3 will be automatic,
# but for me it wasn't, and I want all the speed I can get...
extra_compile_args = ["-O3"]

if sys.platform == "win32":
    extra_compile_args = ["-O2"]

setup(
    # The metadata
    name="Aptus",
    description=doclines[0],
    long_description="\n".join(doclines[2:]),
    long_description_content_type="text/x-rst",
    version=version,
    author="Ned Batchelder",
    author_email="ned@nedbatchelder.com",
    url="http://nedbatchelder.com/code/aptus",
    license="MIT",
    classifiers=list(filter(None, classifiers.split("\n"))),
    python_requires=">=3.9",

    project_urls={
        "Documentation": "https://nedbatchelder.com/code/aptus/v3.html",
        "Code": "http://github.com/nedbat/aptus",
        "Issues": "https://github.com/nedbat/aptus/issues",
        "Funding": "https://github.com/users/nedbat/sponsorship",
    },

    # The data
    packages=[
        "aptus",
        "aptus.gui",
        "aptus.web",
    ],

    package_dir={
        "": "src",
    },

    package_data={
        "aptus": [
            "*.ico",
            "*.png",
            "palettes/*.ggr",
            "web/static/*.*",
            "web/templates/*.*",
        ]
    },

    ext_modules=[
        Extension(
            "aptus.engine",
            sources=["ext/engine.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=extra_compile_args,
        ),
    ],

    entry_points={
        "console_scripts": [
            "aptuscmd = aptus.cmdline:main",
            "aptusgui = aptus.gui:main",
            "aptusweb = aptus.web:main",
        ],
    },

    install_requires=[
        "Pillow",
        "numpy",
    ],

    extras_require={
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

    options=options,
)
