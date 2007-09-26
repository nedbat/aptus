// The Aptus Engine C extension for computing Mandelbrot fractals (hopefully quickly).

#include "Python.h"
#include "numpy/arrayobject.h"

typedef double aptfloat;

typedef struct {
    aptfloat i, r;
} complex_t;

// Global Parameters to the module.
static int max_iter;

static complex_t xy0;
static complex_t xyd;

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
compute_count(int xi, int yi)
{
    int count = 0;
    complex_t c;
    c.r = xy0.r + xi*xyd.r;
    c.i = xy0.i + yi*xyd.i;

    complex_t z = {0,0};
    complex_t znew;
    complex_t z2;
    
    complex_t cycle_check = z;

    int cycle_period = INITIAL_CYCLE_PERIOD;
    int cycle_tries = CYCLE_TRIES;
    int cycle_countdown = cycle_period;

    while (count <= max_iter) {
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

    if (count > max_iter) {
        stats.maxedpoints++;
        count = 0;
    }

    return count;
}

// mandelbrot_point

static char mandelbrot_point_doc[] = "Compute a Mandelbrot count for a point";

static PyObject *
mandelbrot_point(PyObject *self, PyObject *args)
{
    int xi, yi;
    
    if (!PyArg_ParseTuple(args, "ii", &xi, &yi)) {
        return NULL;
    }

    int count = compute_count(xi, yi);

    return Py_BuildValue("i", count);
}

// mandelbrot_array

static char mandelbrot_array_doc[] = "Compute Mandelbrot counts for an array";

static PyObject *
mandelbrot_array(PyObject *self, PyObject *args)
{
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
                c = compute_count(xi, yi);
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
                        c2 = compute_count(curx, cury);
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

// set_geometry

static char set_geometry_doc[] = "Set the geometry, x0, y0, xd, yd";

static PyObject *
set_geometry(PyObject *self, PyObject *args)
{
    double lx0, ly0, lxd, lyd;
    
    if (!PyArg_ParseTuple(args, "dddd", &lx0, &ly0, &lxd, &lyd)) {
        return NULL;
    }
    
    xy0.r = lx0;
    xy0.i = ly0;
    xyd.r = lxd;
    xyd.i = lyd;

    epsilon = xyd.r/2;

    return Py_BuildValue("");
}

// set_maxiter

static char set_maxiter_doc[] = "Set the maximum iteration";

static PyObject *
set_maxiter(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "i", &max_iter)) {
        return NULL;
    }

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

// Module definition

static PyMethodDef
aptus_engine_methods[] = {
    { "mandelbrot_point",   mandelbrot_point,   METH_VARARGS, mandelbrot_point_doc },
    { "mandelbrot_array",   mandelbrot_array,   METH_VARARGS, mandelbrot_array_doc },
    { "set_geometry",       set_geometry,       METH_VARARGS, set_geometry_doc },
    { "set_maxiter",        set_maxiter,        METH_VARARGS, set_maxiter },
    { "set_check_cycles",   set_check_cycles,   METH_VARARGS, set_check_cycles_doc },
    { "clear_stats",        clear_stats,        METH_VARARGS, clear_stats_doc },
    { "get_stats",          get_stats,          METH_VARARGS, get_stats_doc },
    { "float_sizes",        float_sizes,        METH_VARARGS, "Get sizes of float types"},
    { NULL, NULL }
};

void
initaptus_engine(void)
{
    import_array();
    Py_InitModule("aptus_engine", aptus_engine_methods);
}
