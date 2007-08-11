#include "Python.h"

typedef long double float_t;

static int max_iter;

static float_t x0;
static float_t yy0;
static float_t xd;
static float_t yd;

static PyObject *
mandelbrot(PyObject *self, PyObject *args)
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
    
    while (count <= max_iter) {
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
    
    x0 = lx0;
    yy0 = ly0;
    xd = lxd;
    yd = lyd;
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef mandext_methods[] = {
    {"mandelbrot", mandelbrot, METH_VARARGS, "Compute a mandelbrot count for a point"},
    {"set_params", set_params, METH_VARARGS, "Set parameters"},
    {NULL, NULL}
};

void
initmandext(void)
{
    Py_InitModule("mandext", mandext_methods);
}
