# setup.py for Aptus.
"""\
Aptus: A Mandelbrot set explorer and renderer.
"""

import distutils
from distutils.core import setup, Extension

try:
    import numpy
except:
    raise Exception("Need numpy, from http://numpy.scipy.org/")

version = "1.0"

setup(
    name = "Aptus",
    description = "Fast Mandelbrot calculation",
    version = version,
    
    packages = [
        'aptus'
        ],

    package_dir = {
        'aptus': 'aptus'
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
