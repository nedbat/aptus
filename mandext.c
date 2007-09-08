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

struct {
    int     maxiter;    // Max iteration that isn't in the set.
    int     totaliter;  // Total number of iterations.
} stats;

//#define CYCLES 1

#ifdef CYCLES
#define MAX_CYCLE 500

typedef union {
    complex_t c;
    char bytes[sizeof(complex_t)];
} float_cmp;

static float_cmp cycle_buf[MAX_CYCLE];
#endif

static PyObject *
mandelbrot_count(PyObject *self, PyObject *args)
{
    int count = 0;
    
    int xi, yi;
    
    if (!PyArg_ParseTuple(args, "ii", &xi, &yi)) {
        return NULL;
    }

    complex_t c;
    c.r = xy0.r + xi*xyd.r;
    c.i = xy0.i + yi*xyd.i;
    
    complex_t z = {0,0};
    complex_t znew;
    complex_t z2;
    
#ifdef CYCLES
    float_cmp * pCycle = cycle_buf;
    float_cmp * pCheck;
    int cycle = 0;
#endif

    while (count <= max_iter) {
#ifdef CYCLES
        /* Store the current values for orbit checking. */
        if ((pCycle - cycle_buf) < MAX_CYCLE) {
            pCycle->c = z;
            pCycle++;
        }
#endif

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
        
#ifdef CYCLES
        /* Check the orbits. */
        float_cmp cmp;
        cmp.c = z;
        
        for (pCheck = cycle_buf; pCheck < pCycle; pCheck++) {
            if (memcmp(cmp.bytes, pCheck->bytes, sizeof(cmp.bytes)) == 0) {
                cycle = (pCycle - pCheck)/2;
                break;
            }
        }
        
        if (cycle) {
            break;
        }
#endif
    }

    if (count > max_iter) {
        count = 0;
    }

#ifdef CYCLES
    count = cycle;
#endif
    
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
    return Py_BuildValue("");
}

static PyObject *
get_stats(PyObject *self, PyObject *args)
{
    return Py_BuildValue("{sisi}",
        "maxiter", stats.maxiter,
        "totaliter", stats.totaliter
        );        
}

static PyMethodDef mandext_methods[] = {
    {"mandelbrot_count", mandelbrot_count, METH_VARARGS, "Compute a mandelbrot count for a point"},
    {"set_params", set_params, METH_VARARGS, "Set parameters"},
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
