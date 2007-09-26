// The Aptus Engine C extension for computing Mandelbrot fractals (hopefully quickly).

#include "Python.h"
#include "numpy/arrayobject.h"
#include "structmember.h"

// Type definitions.
typedef double aptfloat;

typedef struct {
    aptfloat i, r;
} aptcomplex;

// The Engine type.

typedef struct {
    PyObject_HEAD
    aptcomplex xy0;         // upper-left point (a pair of floats)
    aptcomplex xyd;         // delta per pixel (a pair of floats)
    int maxiter;            // max iteration count.
} AptEngine;

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
        self->xyd.i = 0.001;
        self->xyd.r = 0.001;
        self->maxiter = 999;
    }

    return (PyObject *)self;
}

static int
AptEngine_init(AptEngine *self, PyObject *args, PyObject *kwds)
{
    /*
    PyObject *first=NULL, *last=NULL, *tmp;

    static char *kwlist[] = {"first", "last", "number", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|OOi", kwlist, 
                                      &first, &last, 
                                      &self->number))
        return -1; 

    if (first) {
        tmp = self->first;
        Py_INCREF(first);
        self->first = first;
        Py_XDECREF(tmp);
    }

    if (last) {
        tmp = self->last;
        Py_INCREF(last);
        self->last = last;
        Py_XDECREF(tmp);
    }
    */
    return 0;
}


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

static PyObject *
AptEngine_get_xyd(AptEngine *self, void *closure)
{
    return Py_BuildValue("dd", self->xyd.r, self->xyd.i);
}

static int
AptEngine_set_xyd(AptEngine *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the xyd attribute");
        return -1;
    }
  
    if (!PyArg_ParseTuple(value, "dd", &self->xyd.r, &self->xyd.i)) {
        return -1;
    }

    return 0;
}

// Statistics.

static struct {
    int     maxiter;        // Max iteration that isn't in the set.
    int     totaliter;      // Total number of iterations.
    int     totalcycles;    // Number of cycles detected.
    int     maxitercycle;   // Max iteration that was finally a cycle.
    int     miniter;        // Minimum iteration count.
    int     maxedpoints;    // Number of points that exceeded the maxiter.
} stats;

static int check_for_cycles = 1;

#define INITIAL_CYCLE_PERIOD 7
#define CYCLE_TRIES 10

static aptfloat epsilon;

inline int
fequal(aptfloat a, aptfloat b)
{
    return fabs(a - b) < epsilon;
}

static int
compute_count(AptEngine * self, int xi, int yi)
{
    int count = 0;
    aptcomplex c;
    c.r = self->xy0.r + xi*self->xyd.r;
    c.i = self->xy0.i + yi*self->xyd.i;

    aptcomplex z = {0,0};
    aptcomplex znew;
    aptcomplex z2;
    
    aptcomplex cycle_check = z;

    int cycle_period = INITIAL_CYCLE_PERIOD;
    int cycle_tries = CYCLE_TRIES;
    int cycle_countdown = cycle_period;

    while (count <= self->maxiter) {
        z2.r = z.r * z.r;
        z2.i = z.i * z.i;
        if (z2.r + z2.i > 4.0) {
            if (count > stats.maxiter) {
                stats.maxiter = count;
            }
            if (stats.miniter == 0 || count < stats.miniter) {
                stats.miniter = count;
            }
            break;
        }
        znew.r = z2.r - z2.i + c.r;
        znew.i = 2 * z.i * z.r + c.i;
        z = znew;
        count++;

        stats.totaliter++;

        if (check_for_cycles) {
            // Check for cycles
            if (fequal(z.r, cycle_check.r) && fequal(z.i, cycle_check.i)) {
                // We're in a cycle!
                stats.totalcycles++;
                if (count > stats.maxitercycle) {
                    stats.maxitercycle = count;
                }
                count = 0;
                //count = cycle_period;
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

    if (count > self->maxiter) {
        stats.maxedpoints++;
        count = 0;
    }

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

// mandelbrot_array

static char mandelbrot_array_doc[] = "Compute Mandelbrot counts for an array";

static PyObject *
mandelbrot_array(AptEngine *self, PyObject *args)
{
    printf("xy0: %f, %f\n", self->xy0.r, self->xy0.i);
    printf("xyd: %f, %f\n", self->xyd.r, self->xyd.i);
    PyArrayObject *arr;
    PyObject * progress;
    
    if (!PyArg_ParseTuple(args, "O!O", &PyArray_Type, &arr, &progress)) {
        return NULL;
    }
    
    if (arr == NULL) {
        return NULL;
    }

    if (!PyCallable_Check(progress)) {
        PyErr_SetString(PyExc_TypeError, "progress must be callable");
        return NULL;
    }

    // Allocate structures
    int w = PyArray_DIM(arr, 1);
    int h = PyArray_DIM(arr, 0);
    int num_pixels = 0;
    int num_trace = 0;
    char info[100];
    
    // status is an array of the status of the pixels.
    //  0: hasn't been computed yet.
    //  1: computed, but not filled.
    //  2: computed and filled.
    char * status = malloc(w*h);
    if (status == NULL) {
        PyErr_SetString(PyExc_MemoryError, "couldn't allocate status");
        return NULL;
    }
    memset(status, 0, w*h);

    // points is an array of points on a boundary.
    typedef struct { int x, y; } pt;
    int ptsalloced = 10000;
    pt * points = malloc(sizeof(pt)*ptsalloced);
    int ptsstored = 0;
    
#define STATUS(x,y) status[(y)*w+(x)]
#define COUNTS(x,y) *(npy_uint32 *)PyArray_GETPTR2(arr, (y), (x))
#define DIR_DOWN    0
#define DIR_LEFT    1
#define DIR_UP      2
#define DIR_RIGHT   3

    // Loop the pixels.
    int xi, yi;
    for (yi = 0; yi < h; yi++) {
        for (xi = 0; xi < w; xi++) {
            char s;
            int c;
            
            // Examine the current pixel.
            s = STATUS(xi, yi);
            if (s == 0) {
                c = compute_count(self, xi, yi);
                COUNTS(xi, yi) = c;
                num_pixels++;
                STATUS(xi, yi) = s = 1;
            }
            else {
                c = COUNTS(xi, yi);
            }
            
            // A pixel that's been calculated but not traced needs to be traced.
            if (s == 1) {
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
                            //printf("Upping points to %d\n", ptsalloced*2);
                            pt * newpoints = malloc(sizeof(pt)*ptsalloced*2);
                            if (newpoints == NULL) {
                                goto done;  // Should treat error differently.
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
                
                // If we saved any boundary points, then we flood fill.
                if (ptsstored > 0) {
                    num_trace++;
                    
                    // Flood fill the region. The points list has all the boundary
                    // points, so we only need to fill left and right from each of
                    // those.
                    int pi;
                    for (pi = 0; pi < ptsstored; pi++) {
                        int ptx = points[pi].x;
                        int pty = points[pi].y;
                        curx = ptx;
                        // Fill left.
                        for (;;) {
                            curx--;
                            if (curx < 0) {
                                break;
                            }
                            if (STATUS(curx, pty) != 0) {
                                break;
                            }
                            COUNTS(curx, pty) = c;
                            num_pixels++;
                            STATUS(curx, pty) = 2;
                        }
                        // Fill right.
                        for (;;) {
                            curx++;
                            if (curx > w-1) {
                                break;
                            }
                            if (STATUS(curx, pty) != 0) {
                                break;
                            }
                            COUNTS(curx, pty) = c;
                            num_pixels++;
                            STATUS(curx, pty) = 2;
                        }
                    } // end for points to fill
                    

                    double frac_complete = ((double)num_pixels)/(w*h);
                    sprintf(info, "trace %d", c);
                    PyObject * arglist = Py_BuildValue("(ds)", frac_complete, info);
                    PyObject * result = PyEval_CallObject(progress, arglist);
                    Py_DECREF(arglist);
                    Py_DECREF(result);
                } // end if points
            } // end if needs trace
        } // end for xi

        double frac_complete = ((double)num_pixels)/(w*h);
        sprintf(info, "scan %d", yi+1);
        PyObject * arglist = Py_BuildValue("(ds)", frac_complete, info);
        PyObject * result = PyEval_CallObject(progress, arglist);
        Py_DECREF(arglist);
        Py_DECREF(result);
    } // end for yi
    
    // Clean up.
done:
    free(status);
    free(points);
    
    return Py_BuildValue("");
}

// set_check_cycles

static char set_check_cycles_doc[] = "Set whether to check for cycles or not";

static PyObject *
set_check_cycles(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "i", &check_for_cycles)) {
        return NULL;
    }

    return Py_BuildValue("");    
}

// clear_stats

static char clear_stats_doc[] = "Clear the statistic counters";

static PyObject *
clear_stats(PyObject *self, PyObject *args)
{
    stats.maxiter = 0;
    stats.totaliter = 0;
    stats.totalcycles = 0;
    stats.maxitercycle = 0;
    stats.miniter = 0;
    stats.maxedpoints = 0;
    
    return Py_BuildValue("");
}

// get_stats

static char get_stats_doc[] = "Get the statistics as a dictionary";

static PyObject *
get_stats(PyObject *self, PyObject *args)
{
    return Py_BuildValue("{sisisisisisi}",
        "maxiter", stats.maxiter,
        "totaliter", stats.totaliter,
        "totalcycles", stats.totalcycles,
        "maxitercycle", stats.maxitercycle,
        "miniter", stats.miniter,
        "maxedpoints", stats.maxedpoints
        );        
}

static PyObject *
float_sizes(PyObject *self, PyObject *args)
{
    return Py_BuildValue("ii", sizeof(double), sizeof(aptfloat));
}

// Type definition

static PyMethodDef
AptEngine_methods[] = {
    { "mandelbrot_point",   (PyCFunction) mandelbrot_point,   METH_VARARGS, mandelbrot_point_doc },
    { "mandelbrot_array",   (PyCFunction) mandelbrot_array,   METH_VARARGS, mandelbrot_array_doc },
    { "set_check_cycles",   (PyCFunction) set_check_cycles,   METH_VARARGS, set_check_cycles_doc },
    { "clear_stats",        (PyCFunction) clear_stats,        METH_VARARGS, clear_stats_doc },
    { "get_stats",          (PyCFunction) get_stats,          METH_VARARGS, get_stats_doc },
    { "float_sizes",        (PyCFunction) float_sizes,        METH_VARARGS, "Get sizes of float types"},
    { NULL, NULL }
};

static PyGetSetDef
AptEngine_getsetters[] = {
    { "xy0", (getter)AptEngine_get_xy0, (setter)AptEngine_set_xy0, "Upper-left corner coordinates", NULL },
    { "xyd", (getter)AptEngine_get_xyd, (setter)AptEngine_set_xyd, "Pixel offsets", NULL },
    { NULL }
};

static PyMemberDef
AptEngine_members[] = {
    {"maxiter", T_INT, offsetof(AptEngine, maxiter), 0, "limit on iterations"},
    { NULL }
};

static PyTypeObject AptEngineType = {
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
    { "float_sizes",        float_sizes,        METH_VARARGS, "Get sizes of float types"},
    { NULL, NULL }
};

void
initaptus_engine(void)
{
    import_array();
    
    PyObject* m;

    if (PyType_Ready(&AptEngineType) < 0) {
        return;
    }

    m = Py_InitModule3("aptus_engine", aptus_engine_methods, "Fast Aptusia Mandelbrot engine.");

    if (m == NULL) {
        return;
    }

    Py_INCREF(&AptEngineType);
    PyModule_AddObject(m, "AptEngine", (PyObject *)&AptEngineType);
}
