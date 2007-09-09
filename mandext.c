// The Mandext C extension for computing Mandelbrot fractals (hopefully quickly).

#include "Python.h"

typedef double float_t;

typedef struct {
    float_t i, r;
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
} stats;

static int check_for_cycles = 1;

#define INITIAL_CYCLE_PERIOD 7
#define CYCLE_TRIES 10

static float_t epsilon;

inline int
fequal(float_t a, float_t b)
{
    return fabs(a - b) < epsilon;
}

static PyObject *
mandelbrot_count(PyObject *self, PyObject *args)
{
    int xi, yi;
    
    if (!PyArg_ParseTuple(args, "ii", &xi, &yi)) {
        return NULL;
    }

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
        count = 0;
    }

    return Py_BuildValue("i", count);
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

    return Py_BuildValue("");    
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
    return Py_BuildValue("ii", sizeof(double), sizeof(float_t));
}

static PyObject *
clear_stats(PyObject *self, PyObject *args)
{
    stats.maxiter = 0;
    stats.totaliter = 0;
    stats.totalcycles = 0;
    stats.maxitercycle = 0;
    
    return Py_BuildValue("");
}

static PyObject *
get_stats(PyObject *self, PyObject *args)
{
    return Py_BuildValue("{sisisisi}",
        "maxiter", stats.maxiter,
        "totaliter", stats.totaliter,
        "totalcycles", stats.totalcycles,
        "maxitercycle", stats.maxitercycle
        );        
}

static PyMethodDef
mandext_methods[] = {
    {"mandelbrot_count", mandelbrot_count, METH_VARARGS, "Compute a mandelbrot count for a point"},
    {"set_params", set_params, METH_VARARGS, "Set parameters"},
    {"set_check_cycles", set_check_cycles, METH_VARARGS, "Set more parameters"},
    {"float_sizes", float_sizes, METH_VARARGS, "Get sizes of float types"},
    {"clear_stats", clear_stats, METH_VARARGS, "Clear the statistic counters"},
    {"get_stats", get_stats, METH_VARARGS, "Get the statistics as a dictionary"},
    {NULL, NULL}
};

void
initmandext(void)
{
    Py_InitModule("mandext", mandext_methods);
}
