from aptus_engine import *

import numpy
import time

def do_array(w, h, maxiter):
    set_params(0,0,0,0, maxiter)
    start = time.time()
    counts = numpy.zeros((h, w), dtype=numpy.uint32)
    foo_array(counts)
    return time.time() - start

def do_point(w, h, maxiter):
    set_params(0,0,0,0, maxiter)
    start = time.time()
    counts = numpy.zeros((h, w), dtype=numpy.uint32)
    for yi in xrange(h):
        for xi in xrange(w):
            c = foo_point(xi, -yi)
            counts[yi,xi] = c
    return time.time() - start

def doit(w, h, maxiter):
    a = do_array(w, h, maxiter)
    p = do_point(w, h, maxiter)
    print "%7d: Array %8.3f, Point %8.3f, delta: %.3f sec" % (maxiter, a, p, p-a)
    
doit(500,500,5000)
doit(500,500,10000)
doit(500,500,20000)
doit(500,500,50000)
doit(500,500,100000)
doit(500,500,200000)
doit(500,500,300000)
doit(500,500,400000)
