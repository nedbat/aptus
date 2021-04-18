from aptus.compute import AptusCompute
from aptus.progress import NullProgressReporter
from aptus import settings

import sys
import time

def timeit(args):
    compute = AptusCompute()
    compute.progress = NullProgressReporter()
    compute.size = 5000, 5000

    grandtotal = 0

    if not args:
        case = 'a'
    else:
        case = args[0]

    if case == 'a':
        compute.center = settings.mandelbrot_center
        compute.diam = settings.mandelbrot_diam, settings.mandelbrot_diam
        nruns = 100
    elif case == 'b':
        compute.center = -1.8605327670201655, -1.2705648690517021e-005
        compute.diam = 2.92062690996144e-010, 2.92062690996144e-010
        compute.iter_limit = 99999
        nruns = 20
    elif case == 'c':
        compute.center = -1.0030917862909408, -0.28088298837940889
        compute.diam = 1.3986199517069311e-008, 1.2034636731605979e-008
        compute.iter_limit = 99999
        nruns = 5
    else:
        print("huh?")
        return

    for i in range(nruns):
        compute.clear_results()
        compute.create_mandel()
        start = time.time()
        compute.compute_pixels()
        total = time.time() - start
        print("%.4f" % total)
        grandtotal += total

    print("Average %.5f over %d runs" % (grandtotal/nruns, nruns))


if __name__ == '__main__':
    timeit(sys.argv[1:])
