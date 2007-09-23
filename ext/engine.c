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

static PyObject *
mandelbrot_count(PyObject *self, PyObject *args)
{
    int xi, yi;
    
    if (!PyArg_ParseTuple(args, "ii", &xi, &yi)) {
        return NULL;
    }

    int count = compute_count(xi, yi);

    return Py_BuildValue("i", count);
}
    
static PyObject *
mandelbrot_array(PyObject *self, PyObject *args)
{
    PyArrayObject *arr;
    PyObject * progress;
    PyObject * arglist;
    PyObject * result;
    
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

    int w = PyArray_DIM(arr, 1);
    int h = PyArray_DIM(arr, 0);
    int xi, yi;
    for (yi = 0; yi < h; yi++) {
        for (xi = 0; xi < w; xi++) {
            *(npy_uint32 *)PyArray_GETPTR2(arr, yi, xi) = compute_count(xi, -yi);
        }

        double frac_complete = ((double)yi+1)/h;
        arglist = Py_BuildValue("(d)", frac_complete);
        result = PyEval_CallObject(progress, arglist);
        Py_DECREF(arglist);
        Py_DECREF(result);
    }
    
    return Py_BuildValue("");
}

static PyObject *
set_params(PyObject *self, PyObject *args)
{
    double lx0, ly0, lxd, lyd;
    
    if (!PyArg_ParseTuple(args, "ddddi", &lx0, &ly0, &lxd, &lyd, &max_iter)) {
        return NULL;
    }
    
    xy0.r = lx0;
    xy0.i = ly0;
    xyd.r = lxd;
    xyd.i = lyd;

    epsilon = xyd.r/2;

    return Py_BuildValue("ddddi", (double)xy0.r, (double)xy0.i, (double)xyd.r, (double)xyd.i, max_iter);
}

static PyObject *
set_check_cycles(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "i", &check_for_cycles)) {
        return NULL;
    }

    return Py_BuildValue("");    
}

static PyObject *
float_sizes(PyObject *self, PyObject *args)
{
    return Py_BuildValue("ii", sizeof(double), sizeof(aptfloat));
}

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

static PyMethodDef
mandext_methods[] = {
    {"mandelbrot_count", mandelbrot_count, METH_VARARGS, "Compute a mandelbrot count for a point"},
    {"mandelbrot_array", mandelbrot_array, METH_VARARGS, ""},
    {"set_params", set_params, METH_VARARGS, "Set parameters"},
    {"set_check_cycles", set_check_cycles, METH_VARARGS, "Set more parameters"},
    {"float_sizes", float_sizes, METH_VARARGS, "Get sizes of float types"},
    {"clear_stats", clear_stats, METH_VARARGS, "Clear the statistic counters"},
    {"get_stats", get_stats, METH_VARARGS, "Get the statistics as a dictionary"},
    {NULL, NULL}
};

void
initaptus_engine(void)
{
    import_array();
    Py_InitModule("aptus_engine", mandext_methods);
}
