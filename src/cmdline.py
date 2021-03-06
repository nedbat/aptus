#!/usr/bin/env python

import sys

from PIL import Image

from aptus.compute import AptusCompute
from aptus.progress import IntervalProgressReporter, ConsoleProgressReporter
from aptus.options import AptusOptions


class AptusCmdApp():
    def main(self, args):
        """ The main for the Aptus command-line tool.
        """
        compute = AptusCompute()
        opts = AptusOptions(compute)
        opts.read_args(args)
        compute.create_mandel()

        compute.progress = IntervalProgressReporter(60, ConsoleProgressReporter())
        compute.compute_pixels()
        pix = compute.color_mandel()
        im = Image.fromarray(pix)
        if compute.supersample > 1:
            print("Resampling image...")
            im = im.resize(compute.size, Image.ANTIALIAS)
        compute.write_image(im, compute.outfile)


def main(args):
    AptusCmdApp().main(args)


if __name__ == '__main__':
    main(sys.argv[1:])
