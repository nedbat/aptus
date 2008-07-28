// The Aptus Engine C extension for computing Mandelbrot fractals.
// copyright 2007-2008, Ned Batchelder
// http://nedbatchelder.com/code/aptus

#include "Python.h"
#include "numpy/arrayobject.h"
#include "structmember.h"

// Type definitions.

// A floating-point number.
typedef double aptfloat;

// A complex number.
typedef struct {
    aptfloat r, i;
} aptcomplex;

// Specific-sized integers.
typedef npy_uint8 u1int;
typedef npy_uint32 u4int;
typedef npy_uint64 u8int;

// Macros lifted from Linux kernel.  Use likely(cond) in an if to indicate that
// condition is most likely true, and unlikely(cond) to indicate unlikely to be
// true.  The compiler will arrange the generated code so the straight-line path
// is the likely case.  This helps the CPU pipeline run best.
#ifdef __GNUC__
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)
#else
#define likely(x)       (x)
#define unlikely(x)     (x)
#endif

// The Engine type.

typedef struct {
    PyObject_HEAD
    aptcomplex ri0;         // upper-left point (a pair of floats)
    aptcomplex ridx;        // delta per pixel in x direction (a pair of floats)
    aptcomplex ridy;        // delta per pixel in y direction (a pair of floats)
    aptcomplex rijulia;     // julia point.
    
    int iter_limit;         // limit on iteration count.
    aptfloat bailout;       // escape radius.
    int check_for_cycles;   // should we check for cycles?
    aptfloat epsilon;       // the epsilon to use when checking for cycles.
    aptfloat cont_levels;   // the number of continuous levels to compute.
    int blend_colors;       // how many levels of color should we blend?
    int trace_boundary;     // should we use boundary tracing?
    int julia;              // are we doing julia or mandelbrot?

    // Parameters controlling the cycle detection.
    struct {
        int     initial_period; // save every nth z value as a cycle check.
        int     tries;          // use each period value this many times before choosing a new period.
        int     factor;         // to get a new period, multiply by this,
        int     delta;          //  .. and add this.
    } cycle_params;
    
    // Statistics about the computation.
    struct {
        int     maxiter;        // Max iteration that isn't in the set.
        u8int   totaliter;      // Total number of iterations.
        u4int   totalcycles;    // Number of cycles detected.
        u4int   minitercycle;   // Min iteration that was a cycle.
        u4int   maxitercycle;   // Max iteration that was finally a cycle.
        int     miniter;        // Minimum iteration count.
        u4int   maxedpoints;    // Number of points that exceeded the maxiter.
        u4int   computedpoints; // Number of points that were actually computed.
        u4int   boundaries;     // Number of boundaries traced.
        u4int   boundariesfilled; // Number of boundaries filled.
        u4int   longestboundary; // Most points in a traced boundary.
    } stats;

} AptEngine;

// Class methods

static void
AptEngine_dealloc(AptEngine *self)
{
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
AptEngine_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    AptEngine * self;

    self = (AptEngine *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->ri0.r = 0.0;
        self->ri0.i = 0.0;
        self->ridx.r = 0.001;
        self->ridx.i = 0.0;
        self->ridy.r = 0.0;
        self->ridy.i = 0.001;
        self->rijulia.r = 0.0;
        self->rijulia.i = 0.0;
        self->iter_limit = 999;
        self->bailout = 2.0;
        self->check_for_cycles = 1;
        self->trace_boundary = 1;
        self->cont_levels = 1.0;
        self->blend_colors = 1;
        self->julia = 0;

        self->cycle_params.initial_period = 43;
        self->cycle_params.tries = 10;
        self->cycle_params.factor = 2;
        self->cycle_params.delta = -1;
    }

    return (PyObject *)self;
}

static int
AptEngine_init(AptEngine *self, PyObject *args, PyObject *kwds)
{
    return 0;
}

// ri0 property methods

static PyObject *
get_ri0(AptEngine *self, void *closure)
{
    return Py_BuildValue("dd", self->ri0.r, self->ri0.i);
}

static int
set_ri0(AptEngine *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the ri0 attribute");
        return -1;
    }
  
    if (!PyArg_ParseTuple(value, "dd", &self->ri0.r, &self->ri0.i)) {
        return -1;
    }

    return 0;
}

// ridxdy property methods

static PyObject *
get_ridxdy(AptEngine *self, void *closure)
{
    return Py_BuildValue("dddd", self->ridx.r, self->ridx.i, self->ridy.r, self->ridy.i);
}

static int
set_ridxdy(AptEngine *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the ridxdy attribute");
        return -1;
    }
  
    if (!PyArg_ParseTuple(value, "dddd", &self->ridx.r, &self->ridx.i, &self->ridy.r, &self->ridy.i)) {
        return -1;
    }

    // Make a crude estimate of an epsilon to use for cycle checking.
    self->epsilon = (self->ridx.r+self->ridx.i)/2;
    
    return 0;
}

// rijulia property methods

static PyObject *
get_rijulia(AptEngine *self, void *closure)
{
    return Py_BuildValue("dd", self->rijulia.r, self->rijulia.i);
}

static int
set_rijulia(AptEngine *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the rijulia attribute");
        return -1;
    }
  
    if (!PyArg_ParseTuple(value, "dd", &self->rijulia.r, &self->rijulia.i)) {
        return -1;
    }

    return 0;
}

// cycle_params property methods

static PyObject *
get_cycle_params(AptEngine *self, void *closure)
{
    return Py_BuildValue("iiii",
        self->cycle_params.initial_period,
        self->cycle_params.tries,
        self->cycle_params.factor,
        self->cycle_params.delta
        );
}

static int
set_cycle_params(AptEngine *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the cycle_params attribute");
        return -1;
    }
  
    if (!PyArg_ParseTuple(value, "iiii",
        &self->cycle_params.initial_period,
        &self->cycle_params.tries,
        &self->cycle_params.factor,
        &self->cycle_params.delta
        ))
    {
        return -1;
    }

    return 0;
}

// Are two floating point numbers equal?
inline int
fequal(AptEngine *self, aptfloat a, aptfloat b)
{
    return fabs(a - b) < self->epsilon;
}

// An experiment in leaving out statistics and progress reporting.
// No idea why, but making these empty macros (thereby removing a
// bunch of code) makes the code *slower*.
#define STATS_DECL(x)   x
#define STATS_CODE(x)   x

// compute_count: the heart of the Mandelbrot algorithm.
//
// Given an integer coordinate xi,yi, return the iteration count for that point
// in the current array.

static int
compute_count(AptEngine *self, int xi, int yi)
{
    int count = 0;

    // The complex point we're computing for.
    aptcomplex c;

    // z is the value we're iterating, znew and z2 are intermediates.
    aptcomplex z;
    aptcomplex znew;
    aptcomplex z2;

    if (self->julia) {
        c.r = self->rijulia.r;
        c.i = self->rijulia.i;
        z.r = self->ri0.r + xi*self->ridx.r + yi*self->ridy.r;
        z.i = self->ri0.i + xi*self->ridx.i + yi*self->ridy.i;
    }
    else {
        c.r = self->ri0.r + xi*self->ridx.r + yi*self->ridy.r;
        c.i = self->ri0.i + xi*self->ridx.i + yi*self->ridy.i;
        z.r = 0.0;
        z.i = 0.0;
    }
    
    // Cycle checking bookkeeping variables.
    aptcomplex cycle_check = z;

    int cycle_period = self->cycle_params.initial_period;
    int cycle_tries = self->cycle_params.tries;
    int cycle_countdown = cycle_period;

    aptfloat bail2 = self->bailout * self->bailout;
    
    // Macros for the meat of the iterations, since we need to do them in a few
    // places.
#define ITER1                               \
    z2.r = z.r * z.r;                       \
    z2.i = z.i * z.i;

#define ITER2                               \
    znew.r = z2.r - z2.i + c.r;             \
    znew.i = 2 * z.i * z.r + c.i;           \
    z = znew;                               \
    STATS_CODE(self->stats.totaliter++;)

    // Loop over the iterations.
    while (likely(count <= self->iter_limit)) {
        ITER1;
        if (unlikely(z2.r + z2.i > bail2)) {
            // The point has escaped the bailout.  Update the stats and bail out.
            STATS_CODE(
            if (unlikely(count > self->stats.maxiter)) {
                self->stats.maxiter = count;
            }
            if (unlikely(self->stats.miniter == 0 || count < self->stats.miniter)) {
                self->stats.miniter = count;
            }
            )
            break;
        }
        ITER2;

        if (likely(self->check_for_cycles)) {
            // Check for cycles
            if (unlikely(fequal(self, z.r, cycle_check.r) && fequal(self, z.i, cycle_check.i))) {
                // We're in a cycle! Update stats, and end the iterations.
                STATS_CODE(
                self->stats.totalcycles++;
                if (unlikely(count > self->stats.maxitercycle)) {
                    self->stats.maxitercycle = count;
                }
                if (unlikely(self->stats.minitercycle == 0 || count < self->stats.minitercycle)) {
                    self->stats.minitercycle = count;
                }
                )
                // A cycle means we're inside the set (count of 0).
                count = 0;
                break;
            }
            
            if (--cycle_countdown == 0) {
                // Take a new cycle_check point.
                cycle_check = z;
                cycle_countdown = cycle_period;
                if (--cycle_tries == 0) {
                    cycle_period = self->cycle_params.factor*cycle_period-self->cycle_params.delta;
                    cycle_tries = self->cycle_params.tries;
                }
            }
        }

        count++;
    }

    // Counts above the iteration limit are colored as if they were in the set.
    if (unlikely(count > self->iter_limit)) {
        STATS_CODE(self->stats.maxedpoints++;)
        count = 0;
    }
    
    // Smooth coloring.
    if (count > 0 && self->cont_levels != 1) {
        // http://linas.org/art-gallery/escape/smooth.html

        // three more iterations to reduce the error.
        ITER2;      // we didn't finish the last one.
        ITER1; ITER2;
        ITER1; ITER2;
        ITER1; ITER2;

        // The 2 here is the power of the iteration, not the bailout.
        double delta = log(log(sqrt(z2.r + z2.i)))/log(2);
        double fcount = count + 3 - delta;
        if (unlikely(fcount < 1)) {
            // Way outside the set, continuous mode acts weird.  Cut it off at 1.
            fcount = 1;
        }
        count = fcount * self->cont_levels;
    }
    
    STATS_CODE(self->stats.computedpoints++;)
    
    return count;
}

// mandelbrot_point

static char mandelbrot_point_doc[] = "Compute a Mandelbrot count for a point";

static PyObject *
mandelbrot_point(AptEngine *self, PyObject *args)
{
    int xi, yi;
    
    if (!PyArg_ParseTuple(args, "ii", &xi, &yi)) {
        return NULL;
    }

    int count = compute_count(self, xi, yi);

    return Py_BuildValue("i", count);
}

// Helper: call_progress
STATS_DECL(
static int
call_progress(AptEngine *self, PyObject *progress, double frac_complete, char *info)
{
    int ok = 1;
    PyObject * arglist = Py_BuildValue("(ds)", frac_complete, info);
    PyObject * result = PyEval_CallObject(progress, arglist);
    if (result == NULL) {
        ok = 0;
    }
    Py_DECREF(arglist);
    Py_XDECREF(result);
    return ok;
}
)

// Helper: display a really big number in a portable way

static char *
human_u8int(u8int big, char *buf)
{
    float little = big;
    if (big < 10000000) {   // 10 million
        sprintf(buf, "%lu", (u4int)big);
    }
    else if (big < 1000000000) {    // 1 billion
        little /= 1e6;
        sprintf(buf, "%.1fM", little);
    }
    else {
        little /= 1e9;
        sprintf(buf, "%.1fB", little);
    }
    
    return buf;
}

// mandelbrot_array

static char mandelbrot_array_doc[] = "Compute Mandelbrot counts for an array";

static PyObject *
mandelbrot_array(AptEngine *self, PyObject *args)
{
    // Arguments to the function.
    PyArrayObject *counts;
    // status is an array of the status of the pixels.
    //  0: hasn't been computed yet.
    //  1: computed, but not filled.
    //  2: computed and filled.
    PyArrayObject *status;
    PyObject * progress;
    
    // Malloc'ed buffers.
    typedef struct { int x, y; } pt;
    pt * points = NULL;
    
    int ok = 0;
    
    if (!PyArg_ParseTuple(args, "O!O!O", &PyArray_Type, &counts, &PyArray_Type, &status, &progress)) {
        goto done;
    }
    
    if (!PyCallable_Check(progress)) {
        PyErr_SetString(PyExc_TypeError, "progress must be callable");
        goto done;
    }

    // Allocate structures
    int w = PyArray_DIM(counts, 1);
    int h = PyArray_DIM(counts, 0);
    int num_pixels = 0;
    
    // points is an array of points on a boundary.
    int ptsalloced = 10000;
    points = malloc(sizeof(pt)*ptsalloced);
    int ptsstored = 0;

    STATS_DECL(    
    // Progress reporting stuff.
    char info[100];
    char uinfo[100];
    u8int last_progress = 0;    // the totaliter the last time we called the progress function.
    const int MIN_PROGRESS = 1000000;  // Don't call progress unless we've done this many iters.
    )

#define STATUS(x,y) *(npy_uint8 *)PyArray_GETPTR2(status, (y), (x))
#define COUNTS(x,y) *(npy_uint32 *)PyArray_GETPTR2(counts, (y), (x))
#define DIR_DOWN    0
#define DIR_LEFT    1
#define DIR_UP      2
#define DIR_RIGHT   3

    // Loop the pixels.
    int xi, yi;
    u1int s;
    int c = 0;

    for (yi = 0; yi < h; yi++) {
        for (xi = 0; xi < w; xi++) {
            // Examine the current pixel.
            s = STATUS(xi, yi);
            if (s == 0) {
                c = compute_count(self, xi, yi);
                COUNTS(xi, yi) = c;
                num_pixels++;
                STATUS(xi, yi) = s = 1;
            }
            else if (s == 1 && self->trace_boundary) {
                c = COUNTS(xi, yi);
            }
            
            // A pixel that's been calculated but not traced needs to be traced.
            if (s == 1 && self->trace_boundary) {
                char curdir = DIR_DOWN;
                int curx = xi, cury = yi;
                int origx = xi, origy = yi;
                int lastx = xi, lasty = yi;
                int start = 1;
                
                STATUS(xi, yi) = 2;
                ptsstored = 0;
                
                // Walk the boundary
                for (;;) {
                    // Eventually, we reach our starting point. Stop.
                    if (unlikely(!start && curx == origx && cury == origy && curdir == DIR_DOWN)) {
                        break;
                    }
                    
                    // Move to the next position. If we're off the field, turn left.
                    switch (curdir) {
                    case DIR_DOWN:
                        if (unlikely(cury >= h-1)) {
                            curdir = DIR_RIGHT;
                            continue;
                        }
                        cury++;
                        break;

                    case DIR_LEFT:
                        if (unlikely(curx <= 0)) {
                            curdir = DIR_DOWN;
                            continue;
                        }
                        curx--;
                        break;
                    
                    case DIR_UP:
                        if (unlikely(cury <= 0)) {
                            curdir = DIR_LEFT;
                            continue;
                        }
                        cury--;
                        break;
                    
                    case DIR_RIGHT:
                        if (unlikely(curx >= w-1)) {
                            curdir = DIR_UP;
                            continue;
                        }
                        curx++;
                        break;
                    }
                    
                    // Get the count of the next position.
                    int c2;
                    if (STATUS(curx, cury) == 0) {
                        c2 = compute_count(self, curx, cury);
                        COUNTS(curx, cury) = c2;
                        num_pixels++;
                        STATUS(curx, cury) = 1;
                    }
                    else {
                        c2 = COUNTS(curx, cury);
                    }
                    
                    // If the same color, turn right, else turn left.
                    if (c2 == c) {
                        STATUS(curx, cury) = 2;
                        // Append the point to the points list, growing dynamically
                        // if we have to.
                        if (unlikely(ptsstored == ptsalloced)) {
                            pt * newpoints = malloc(sizeof(pt)*ptsalloced*2);
                            if (newpoints == NULL) {
                                PyErr_SetString(PyExc_MemoryError, "couldn't allocate points");
                                goto done;
                            }
                            memcpy(newpoints, points, sizeof(pt)*ptsalloced);
                            ptsalloced *= 2;
                            free(points);
                            points = newpoints;
                        }
                        points[ptsstored].x = curx;
                        points[ptsstored].y = cury;
                        ptsstored++;
                        lastx = curx;
                        lasty = cury;
                        curdir = (curdir+1) % 4;    // Turn right
                    }
                    else {
                        curx = lastx;
                        cury = lasty;
                        curdir = (curdir+3) % 4;    // Turn left
                    }
                    
                    start = 0;
                } // end for boundary points
                
                STATS_CODE(
                self->stats.boundaries++;
                if (ptsstored > self->stats.longestboundary) {
                    self->stats.longestboundary = ptsstored;
                }
                )
                
                // If we saved enough boundary points, then we flood fill. The
                // points are orthogonally connected, so we need at least eight
                // to enclose a fillable point.
                if (unlikely(ptsstored >= 8)) {
                    // Flood fill the region. The points list has all the boundary
                    // points, so we only need to fill left from each of those.
                    int num_filled = 0;
                    int pi;
                    for (pi = 0; pi < ptsstored; pi++) {
                        int ptx = points[pi].x;
                        int pty = points[pi].y;
                        // Fill left.
                        for (;;) {
                            ptx--;
                            if (ptx < 0) {
                                break;
                            }
                            if (STATUS(ptx, pty) != 0) {
                                break;
                            }
                            COUNTS(ptx, pty) = c;
                            num_pixels++;
                            num_filled++;
                            STATUS(ptx, pty) = 2;
                        }
                    } // end for points to fill
                
                    STATS_CODE(    
                    if (num_filled > 0) {
                        self->stats.boundariesfilled++;

                        // If this was a large boundary, call the progress function.
                        if (ptsstored > w) {
                            if (self->stats.totaliter - last_progress > MIN_PROGRESS) {
                                sprintf(info, "trace %d * %d, totaliter %s", c, ptsstored, human_u8int(self->stats.totaliter, uinfo));
                                if (!call_progress(self, progress, ((double)num_pixels)/(w*h), info)) {
                                    goto done;
                                }
                                last_progress = self->stats.totaliter;
                            }
                        }
                    }
                    )
                } // end if points
            } // end if needs trace
        } // end for xi

        STATS_CODE(
        // At the end of the scan line, call progress if we've made enough progress
        if (self->stats.totaliter - last_progress > MIN_PROGRESS) {
            sprintf(info, "scan %d, totaliter %s", yi+1, human_u8int(self->stats.totaliter, uinfo));
            if (!call_progress(self, progress, ((double)num_pixels)/(w*h), info)) {
                goto done;
            }
            last_progress = self->stats.totaliter;
        }
        )
    } // end for yi
    
    // Clean up.
    ok = 1;
    
done:
    if (points != NULL) {
        free(points);
    }
    
    return ok ? Py_BuildValue("") : NULL;
}

// apply_palette

static char apply_palette_doc[] = "Color an array based on counts and palette";

static PyObject *
apply_palette(AptEngine *self, PyObject *args)
{
    // Arguments to the function.
    PyArrayObject *counts;
    PyObject * colbytes_obj;
    PyObject * incolor_obj;
    PyArrayObject *pix;
    int phase;
    double scale;
    
    // Objects we get during the function.
    PyObject * pint = NULL;
    int ok = 0;
    
    if (!PyArg_ParseTuple(args, "O!OidOO!", &PyArray_Type, &counts, &colbytes_obj, &phase, &scale, &incolor_obj, &PyArray_Type, &pix)) {
        goto done;
    }
    
    // Unpack the palette a bit.
    u1int * colbytes;
    Py_ssize_t ncolbytes;
    if (PyString_AsStringAndSize(colbytes_obj, (char**)&colbytes, &ncolbytes) < 0) {
        goto done;
    }
    int ncolors = ncolbytes / 3;

    u1int incolbytes[3];
    int i;
    for (i = 0; i < 3; i++) {
        pint = PySequence_GetItem(incolor_obj, i);
        incolbytes[i] = (u1int)PyInt_AsLong(pint);
        Py_CLEAR(pint);
    }
    
    // A one-element cache of count and color.
    npy_uint8 *plastpix = NULL;
    npy_uint32 lastc = (*(npy_uint32 *)PyArray_GETPTR2(counts, 0, 0))+1;    // Something different than the first value.
    
    // Walk the arrays
    int h = PyArray_DIM(counts, 0);
    int w = PyArray_DIM(counts, 1);
    int count_stride = PyArray_STRIDE(counts, 1);
    int pix_stride = PyArray_STRIDE(pix, 1);
    int x, y;
    for (y = 0; y < h; y++) {
        // The count for this pixel.
        void *pcount = PyArray_GETPTR2(counts, y, 0);
        // The pointer to the pixel's RGB bytes.
        npy_uint8 *ppix = (npy_uint8 *)PyArray_GETPTR3(pix, y, 0, 0);

        for (x = 0; x < w; x++) {
            npy_uint32 c = *(npy_uint32*)pcount;
            if (c == lastc) {
                // A hit on the one-element cache. Copy the old value.
                memcpy(ppix, plastpix, 3);
            }
            else {
                if (c > 0) {
                    // The pixel is outside the set, color it with the palette.
                    if (self->blend_colors == 1) {
                        // Not blending colors, each count is a literal palette
                        // index.
                        int cindex = (c + phase) % ncolors;
                        memcpy(ppix, (colbytes + cindex*3), 3);
                    }
                    else {
                        double cf = c * scale / self->blend_colors;
                        int cbase = cf;
                        float cfrac = cf - cbase;
                        int c1index = (cbase + phase) % ncolors;
                        int c2index = (cbase + 1 + phase) % ncolors;
                        for (i = 0; i < 3; i++) {
                            float col1 = colbytes[c1index*3+i];
                            float col2 = colbytes[c2index*3+i];
                            ppix[i] = (int)(col1 + (col2-col1)*cfrac);
                        }
                    }
                }
                else {
                    // The pixel is in the set, color it with the incolor.
                    memcpy(ppix, incolbytes, 3);
                }
                
                // Save this value in our one-element cache.
                lastc = c;
                plastpix = ppix;
            }
            
            // Advance to the next pixel.
            pcount += count_stride;
            ppix += pix_stride;
        }
    }

    ok = 1;
    
done:
    Py_XDECREF(pint);
    
    return ok ? Py_BuildValue("") : NULL;
}

// clear_stats

static char clear_stats_doc[] = "Clear the statistic counters";

static PyObject *
clear_stats(AptEngine *self)
{
    self->stats.maxiter = 0;
    self->stats.totaliter = 0;
    self->stats.totalcycles = 0;
    self->stats.minitercycle = 0;
    self->stats.maxitercycle = 0;
    self->stats.miniter = 0;
    self->stats.maxedpoints = 0;
    self->stats.computedpoints = 0;
    self->stats.boundaries = 0;
    self->stats.boundariesfilled = 0;
    self->stats.longestboundary = 0;

    return Py_BuildValue("");
}

// get_stats

static char get_stats_doc[] = "Get the statistics as a dictionary";

static PyObject *
get_stats(AptEngine *self)
{
    return Py_BuildValue("{sisKsIsIsIsisIsIsIsIsI}",
        "maxiter", self->stats.maxiter,
        "totaliter", self->stats.totaliter,
        "totalcycles", self->stats.totalcycles,
        "minitercycle", self->stats.minitercycle,
        "maxitercycle", self->stats.maxitercycle,
        "miniter", self->stats.miniter,
        "maxedpoints", self->stats.maxedpoints,
        "computedpoints", self->stats.computedpoints,
        "boundaries", self->stats.boundaries,
        "boundariesfilled", self->stats.boundariesfilled,
        "longestboundary", self->stats.longestboundary
        );
}

// type_check

static char type_check_doc[] = "Try out types in the C extension";

static PyObject *
type_check(PyObject *self, PyObject *args)
{
    char info[200];
    char uinfo[200];
    u8int big = 1;
    big <<= 40;
    sprintf(info, "Big 1<<40 = %s", human_u8int(big, uinfo));
    
    return Py_BuildValue("{sisisisiss}",
        "double", sizeof(double),
        "aptfloat", sizeof(aptfloat),
        "u4int", sizeof(u4int),
        "u8int", sizeof(u8int),
        "sprintf", info
        );
}

// Type definition

static PyMemberDef
AptEngine_members[] = {
    { "iter_limit",     T_INT,      offsetof(AptEngine, iter_limit),        0, "Limit on iterations" },
    { "bailout",        T_DOUBLE,   offsetof(AptEngine, bailout),           0, "Radius of the escape circle" },
    { "cont_levels",    T_DOUBLE,   offsetof(AptEngine, cont_levels),       0, "Number of fractional levels to compute" },
    { "blend_colors",   T_INT,      offsetof(AptEngine, blend_colors),      0, "How many levels of color to blend" },
    { "trace_boundary", T_INT,      offsetof(AptEngine, trace_boundary),    0, "Control whether boundaries are traced" },
    { "julia",          T_INT,      offsetof(AptEngine, julia),             0, "Compute Julia set?" },
    { NULL }
};

static PyGetSetDef
AptEngine_getsetters[] = {
    { "ri0",            (getter)get_ri0,            (setter)set_ri0,            "Upper-left corner coordinates", NULL },
    { "ridxdy",         (getter)get_ridxdy,         (setter)set_ridxdy,         "Pixel offsets", NULL },
    { "rijulia",        (getter)get_rijulia,        (setter)set_rijulia,        "Julia point", NULL },
    { "cycle_params",   (getter)get_cycle_params,   (setter)set_cycle_params,   "Cycle detection parameters", NULL },
    { NULL }
};

static PyMethodDef
AptEngine_methods[] = {
    { "mandelbrot_point",   (PyCFunction) mandelbrot_point,   METH_VARARGS, mandelbrot_point_doc },
    { "mandelbrot_array",   (PyCFunction) mandelbrot_array,   METH_VARARGS, mandelbrot_array_doc },
    { "apply_palette",      (PyCFunction) apply_palette,      METH_VARARGS, apply_palette_doc },
    { "clear_stats",        (PyCFunction) clear_stats,        METH_NOARGS,  clear_stats_doc },
    { "get_stats",          (PyCFunction) get_stats,          METH_NOARGS,  get_stats_doc },
    { NULL }
};

static PyTypeObject
AptEngineType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "AptEngine.AptEngine",     /*tp_name*/
    sizeof(AptEngine),         /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)AptEngine_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "AptEngine objects",       /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    AptEngine_methods,         /* tp_methods */
    AptEngine_members,         /* tp_members */
    AptEngine_getsetters,      /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)AptEngine_init,  /* tp_init */
    0,                         /* tp_alloc */
    AptEngine_new,             /* tp_new */
};


// Module definition

static PyMethodDef
AptEngine_classmethods[] = {
    { "type_check", type_check, METH_VARARGS, type_check_doc },
    { NULL, NULL }
};

void
initengine(void)
{
    import_array();
    
    PyObject* m;

    if (PyType_Ready(&AptEngineType) < 0) {
        return;
    }

    m = Py_InitModule3("aptus.engine", AptEngine_classmethods, "Fast Aptus Mandelbrot engine.");

    if (m == NULL) {
        return;
    }

    Py_INCREF(&AptEngineType);
    PyModule_AddObject(m, "AptEngine", (PyObject *)&AptEngineType);
}
