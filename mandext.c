#include "Python.h"

typedef long double float_t;

static int max_iter;

static PyObject *
mandelbrot(PyObject *self, PyObject *args)
{
    float_t cr, ci;
    int count = 0;
    
    double crd, cid;
    
    if (!PyArg_ParseTuple(args, "dd", &crd, &cid))
        return NULL;

    cr = crd;
    ci = cid;
    
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
        Py_INCREF(Py_None);
        return Py_None;
    }
    else {
        return Py_BuildValue("i", count);
    }
}	

static PyObject *
set_maxiter(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "i", &max_iter))
        return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef mandext_methods[] = {
    {"mandelbrot", mandelbrot, METH_VARARGS, "Compute a mandelbrot count for a point"},
    {"set_maxiter", set_maxiter, METH_VARARGS, "Set the maximum iteration count"},
    {NULL, NULL}
};

void
initmandext(void)
{
    Py_InitModule("mandext", mandext_methods);
}
