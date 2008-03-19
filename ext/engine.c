// The Aptus Engine C extension for computing Mandelbrot fractals (hopefully quickly).

#include "Python.h"
#include "numpy/arrayobject.h"
#include "structmember.h"

// Type definitions.

// A floating-point number.
typedef double aptfloat;

// A complex number.
typedef struct {
    aptfloat i, r;
} aptcomplex;

// Specific-sized integers.
typedef npy_uint8 u1int;
typedef npy_uint32 u4int;
typedef npy_uint64 u8int;

// The Engine type.

typedef struct {
    PyObject_HEAD
    aptcomplex xy0;         // upper-left point (a pair of floats)
    aptcomplex xydx;        // delta per pixel in x direction (a pair of floats)
    aptcomplex xydy;        // delta per pixel in y direction (a pair of floats)
    
    int iter_limit;         // limit on iteration count.
    aptfloat bailout;       // escape radius.
    int check_for_cycles;   // should we check for cycles?
    aptfloat epsilon;       // the epsilon to use when checking for cycles.
    aptfloat cont_levels;   // the number of continuous levels to compute.
    int blend_colors;       // how many levels of color should we blend?
    int trace_boundary;     // should we use boundary tracing?
    
    struct {
        int     maxiter;        // Max iteration that isn't in the set.
        u8int   totaliter;      // Total number of iterations.
        u4int   totalcycles;    // Number of cycles detected.
        u4int   maxitercycle;   // Max iteration that was finally a cycle.
        int     miniter;        // Minimum iteration count.
        u4int   maxedpoints;    // Number of points that exceeded the maxiter.
        u4int   computedpoints; // Number of points that were actually computed.
        u4int   boundaries;     // Number of boundaries traced.
        u4int   boundariesfilled; // Number of boundaries filled.
    } stats;

} AptEngine;

// Class methods

static void
AptEngine_dealloc(AptEngine * self)
{
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *
AptEngine_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    AptEngine * self;

    self = (AptEngine *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->xy0.i = 0.0;
        self->xy0.r = 0.0;
        self->xydx.i = 0.0;
        self->xydx.r = 0.001;
        self->xydy.i = 0.001;
        self->xydy.r = 0.0;
        self->iter_limit = 999;
        self->bailout = 2.0;
        self->check_for_cycles = 1;
        self->trace_boundary = 1;
        self->cont_levels = 1.0;
        self->blend_colors = 1;
    }

    return (PyObject *)self;
}

static int
AptEngine_init(AptEngine *self, PyObject *args, PyObject *kwds)
{
    return 0;
}

// xy0 property methods

static PyObject *
AptEngine_get_xy0(AptEngine *self, void *closure)
{
    return Py_BuildValue("dd", self->xy0.r, self->xy0.i);
}

static int
AptEngine_set_xy0(AptEngine *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the xy0 attribute");
        return -1;
    }
  
    if (!PyArg_ParseTuple(value, "dd", &self->xy0.r, &self->xy0.i)) {
        return -1;
    }

    return 0;
}

// xydxdy property methods

static PyObject *
AptEngine_get_xydxdy(AptEngine *self, void *closure)
{
    return Py_BuildValue("dddd", self->xydx.r, self->xydx.i, self->xydy.r, self->xydy.i);
}

static int
AptEngine_set_xydxdy(AptEngine *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the xydxdy attribute");
        return -1;
    }
  
    if (!PyArg_ParseTuple(value, "dddd", &self->xydx.r, &self->xydx.i, &self->xydy.r, &self->xydy.i)) {
        return -1;
    }

    // Make a crude estimate of an epsilon to use for cycle checking.
    self->epsilon = (self->xydx.r+self->xydx.i)/2;
    
    return 0;
}

// Are two floating point numbers equal?
inline int
fequal(AptEngine * self, aptfloat a, aptfloat b)
{
    return fabs(a - b) < self->epsilon;
}

// compute_count: the heart of the Mandelbrot algorithm.
//
// Given an integer coordinate xi,yi, return the iteration count for that point
// in the current array.

#define INITIAL_CYCLE_PERIOD 7
#define CYCLE_TRIES 10

static int
compute_count(AptEngine * self, int xi, int yi)
{
    int count = 0;

    // The complex point we're computing for.
    aptcomplex c;
    c.r = self->xy0.r + xi*self->xydx.r + yi*self->xydy.r;
    c.i = self->xy0.i + xi*self->xydx.i + yi*self->xydy.i;

    // z is the value we're iterating, znew and z2 are intermediates.
    aptcomplex z = {0,0};
    aptcomplex znew;
    aptcomplex z2;
    
    // Cycle checking bookkeeping variables.
    aptcomplex cycle_check = z;

    int cycle_period = INITIAL_CYCLE_PERIOD;
    int cycle_tries = CYCLE_TRIES;
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
    self->stats.totaliter++;

    // Loop over the iterations.
    while (count <= self->iter_limit) {
        ITER1;
        if (z2.r + z2.i > bail2) {
            // The point has escaped the bailout.  Update the stats and bail out.
            if (count > self->stats.maxiter) {
                self->stats.maxiter = count;
            }
            if (self->stats.miniter == 0 || count < self->stats.miniter) {
                self->stats.miniter = count;
            }
            break;
        }
        ITER2;
        count++;

        if (self->check_for_cycles) {
            // Check for cycles
            if (fequal(self, z.r, cycle_check.r) && fequal(self, z.i, cycle_check.i)) {
                // We're in a cycle! Update stats, and end the iterations.
                self->stats.totalcycles++;
                if (count > self->stats.maxitercycle) {
                    self->stats.maxitercycle = count;
                }
                // A cycle means we're inside the set (count of 0).
                count = 0;
                break;
            }
            
            if (--cycle_countdown == 0) {
                // Take a new cycle_check point.
                cycle_check = z;
                cycle_countdown = cycle_period;
                if (--cycle_tries == 0) {
                    cycle_period = 2*cycle_period-1;
                    cycle_tries = CYCLE_TRIES;
                }
            }
        }
    }

    // Counts above the iteration limit are colored as if they were in the set.
    if (count > self->iter_limit) {
        self->stats.maxedpoints++;
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
        count = fcount * self->cont_levels;
    }
    
    self->stats.computedpoints++;
    
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

static int
call_progress(AptEngine * self, PyObject * progress, double frac_complete, char * info)
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

// Helper: display a really big number in a portable way

static char *
human_u8int(u8int big, char * buf)
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
    PyArrayObject *arr;
    PyObject * progress;
    
    // Malloc'ed buffers.
    typedef struct { int x, y; } pt;
    u1int * status = NULL;
    pt * points = NULL;
    
    int ok = 0;
    
    if (!PyArg_ParseTuple(args, "O!O", &PyArray_Type, &arr, &progress)) {
        goto done;
    }
    
    if (!PyCallable_Check(progress)) {
        PyErr_SetString(PyExc_TypeError, "progress must be callable");
        goto done;
    }

    // Allocate structures
    int w = PyArray_DIM(arr, 1);
    int h = PyArray_DIM(arr, 0);
    int num_pixels = 0;
    
    // status is an array of the status of the pixels.
    //  0: hasn't been computed yet.
    //  1: computed, but not filled.
    //  2: computed and filled.
    status = (u1int *) malloc(w*h);
    if (status == NULL) {
        PyErr_SetString(PyExc_MemoryError, "couldn't allocate status");
        goto done;
    }
    memset(status, 0, w*h);

    // points is an array of points on a boundary.
    int ptsalloced = 10000;
    points = malloc(sizeof(pt)*ptsalloced);
    int ptsstored = 0;
    
    char info[100];
    char uinfo[100];
    
#define STATUS(x,y) status[(y)*w+(x)]
#define COUNTS(x,y) *(npy_uint32 *)PyArray_GETPTR2(arr, (y), (x))
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
                    if (!start && curx == origx && cury == origy && curdir == DIR_DOWN) {
                        break;
                    }
                    
                    // Move to the next position. If we're off the field, turn left.
                    switch (curdir) {
                    case DIR_DOWN:
                        if (cury >= h-1) {
                            curdir = DIR_RIGHT;
                            continue;
                        }
                        cury++;
                        break;

                    case DIR_LEFT:
                        if (curx <= 0) {
                            curdir = DIR_DOWN;
                            continue;
                        }
                        curx--;
                        break;
                    
                    case DIR_UP:
                        if (cury <= 0) {
                            curdir = DIR_LEFT;
                            continue;
                        }
                        cury--;
                        break;
                    
                    case DIR_RIGHT:
                        if (curx >= w-1) {
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
                        if (ptsstored == ptsalloced) {
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
                
                self->stats.boundaries++;
                
                // If we saved enough boundary points, then we flood fill. The
                // points are orthogonally connected, so we need at least eight
                // to enclose a fillable point.
                if (ptsstored >= 8) {
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
                    
                    if (num_filled > 0) {
                        self->stats.boundariesfilled++;

                        // If this was a large boundary, call the progress function.
                        if (ptsstored > w) {
                            sprintf(info, "trace %d * %d, totaliter %s", c, ptsstored, human_u8int(self->stats.totaliter, uinfo));
                            if (!call_progress(self, progress, ((double)num_pixels)/(w*h), info)) {
                                goto done;
                            }
                        }
                    }
                } // end if points
            } // end if needs trace
        } // end for xi

        sprintf(info, "scan %d, totaliter %s", yi+1, human_u8int(self->stats.totaliter, uinfo));
        if (!call_progress(self, progress, ((double)num_pixels)/(w*h), info)) {
            goto done;
        }
    } // end for yi
    
    // Clean up.
    ok = 1;
    
done:
    if (status != NULL) {
        free(status);
    }
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
    
    // Walk the arrays
    int w = PyArray_DIM(counts, 0);
    int h = PyArray_DIM(counts, 1);
    int x, y;
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            // The count for this pixel.
            npy_uint32 c = *(npy_uint32 *)PyArray_GETPTR2(counts, x, y);
            // The pointer to the pixels RGB bytes.
            npy_uint8 *ppix = (npy_uint8 *)PyArray_GETPTR3(pix, x, y, 0);
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
    self->stats.maxitercycle = 0;
    self->stats.miniter = 0;
    self->stats.maxedpoints = 0;
    self->stats.computedpoints = 0;

    return Py_BuildValue("");
}

// get_stats

static char get_stats_doc[] = "Get the statistics as a dictionary";

static PyObject *
get_stats(AptEngine *self, PyObject *args)
{
    return Py_BuildValue("{sisKsIsIsisIsIsIsI}",
        "maxiter", self->stats.maxiter,
        "totaliter", self->stats.totaliter,
        "totalcycles", self->stats.totalcycles,
        "maxitercycle", self->stats.maxitercycle,
        "miniter", self->stats.miniter,
        "maxedpoints", self->stats.maxedpoints,
        "computedpoints", self->stats.computedpoints,
        "boundaries", self->stats.boundaries,
        "boundariesfilled", self->stats.boundariesfilled
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
    { "iter_limit", T_INT, offsetof(AptEngine, iter_limit), 0, "Limit on iterations" },
    { "bailout", T_DOUBLE, offsetof(AptEngine, bailout), 0, "Radius of the escape circle" },
    { "cont_levels", T_DOUBLE, offsetof(AptEngine, cont_levels), 0, "Number of fractional levels to compute" },
    { "blend_colors", T_INT, offsetof(AptEngine, blend_colors), 0, "How many levels of color to blend" },
    { "trace_boundary", T_INT, offsetof(AptEngine, trace_boundary), 0, "Control whether boundaries are traced" },
    { NULL }
};

static PyGetSetDef
AptEngine_getsetters[] = {
    { "xy0", (getter)AptEngine_get_xy0, (setter)AptEngine_set_xy0, "Upper-left corner coordinates", NULL },
    { "xydxdy", (getter)AptEngine_get_xydxdy, (setter)AptEngine_set_xydxdy, "Pixel offsets", NULL },
    { NULL }
};

static PyMethodDef
AptEngine_methods[] = {
    { "mandelbrot_point",   (PyCFunction) mandelbrot_point,   METH_VARARGS, mandelbrot_point_doc },
    { "mandelbrot_array",   (PyCFunction) mandelbrot_array,   METH_VARARGS, mandelbrot_array_doc },
    { "apply_palette",      (PyCFunction) apply_palette,      METH_VARARGS, apply_palette_doc },
    { "clear_stats",        (PyCFunction) clear_stats,        METH_NOARGS,  clear_stats_doc },
    { "get_stats",          (PyCFunction) get_stats,          METH_VARARGS, get_stats_doc },
    { NULL, NULL }
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
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
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
aptus_engine_methods[] = {
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

    m = Py_InitModule3("aptus.engine", aptus_engine_methods, "Fast Aptus Mandelbrot engine.");

    if (m == NULL) {
        return;
    }

    Py_INCREF(&AptEngineType);
    PyModule_AddObject(m, "AptEngine", (PyObject *)&AptEngineType);
}
