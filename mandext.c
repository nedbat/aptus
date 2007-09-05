#include "Python.h"

typedef double float_t;

static int max_iter;

static float_t x0;
static float_t yy0;
static float_t xd;
static float_t yd;

#define CYCLES 1

#ifdef CYCLES
#define MAX_CYCLE 500

typedef union {
    struct {
        float_t a;
        float_t b;
    };
    char bytes[sizeof(float_t)*2];
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
    
    float_t cr = x0 + xi*xd;
    float_t ci = yy0 + yi*yd;
    
    float_t az = 0.0;
    float_t bz = 0.0;
    float_t anew, bnew;
    float_t a2, b2;
    
#ifdef CYCLES
    float_cmp * pCycle = cycle_buf;
    float_cmp * pCheck;
    int cycle = 0;
#endif

    while (count <= max_iter) {
#ifdef CYCLES
        /* Store the current values for orbit checking. */
        if ((pCycle - cycle_buf) < MAX_CYCLE) {
            pCycle->a = az;
            pCycle->b = bz;
            pCycle++;
        }
#endif

        a2 = az * az;
        b2 = bz * bz;
        if (a2 + b2 > 4) {
            break;
        }
        anew = a2 - b2 + cr;
        bnew = 2 * az * bz + ci;
        az = anew;
        bz = bnew;
        count++;

#ifdef CYCLES
        /* Check the orbits. */
        float_cmp cmp;
        cmp.a = az;
        cmp.b = bz;
        
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
    
    x0 = lx0;
    yy0 = ly0;
    xd = lxd;
    yd = lyd;

    return Py_BuildValue("");    
}

static PyObject *
float_sizes(PyObject *self, PyObject *args)
{
    return Py_BuildValue("ii", sizeof(double), sizeof(float_t));
}

static PyMethodDef mandext_methods[] = {
    {"mandelbrot_count", mandelbrot_count, METH_VARARGS, "Compute a mandelbrot count for a point"},
    {"set_params", set_params, METH_VARARGS, "Set parameters"},
    {"float_sizes", float_sizes, METH_VARARGS, "Get sizes of float types"},
    {NULL, NULL}
};

void
initmandext(void)
{
    Py_InitModule("mandext", mandext_methods);
}
