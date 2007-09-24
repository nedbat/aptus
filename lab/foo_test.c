
static int
foo_count(int xi, int yi)
{
    aptfloat f = 1.0;
    
    int i;
    for (i = 0; i < max_iter; i++) {
        f *= 1.0000001;
    }
    
    return 27;
}

static PyObject *
foo_point(PyObject *self, PyObject *args)
{
    int xi, yi;
    
    if (!PyArg_ParseTuple(args, "ii", &xi, &yi)) {
        return NULL;
    }

    int count = foo_count(xi, yi);

    return Py_BuildValue("i", count);
}
    
static PyObject *
foo_array(PyObject *self, PyObject *args)
{
    PyArrayObject *arr;
    
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr)) {
        return NULL;
    }
    
    if (arr == NULL) {
        return NULL;
    }

    int w = PyArray_DIM(arr, 1);
    int h = PyArray_DIM(arr, 0);
    int xi, yi;
    for (yi = 0; yi < h; yi++) {
        for (xi = 0; xi < w; xi++) {
            *(npy_uint32 *)PyArray_GETPTR2(arr, yi, xi) = foo_count(xi, -yi);
        }
    }
    
    return Py_BuildValue("");
}

static PyMethodDef
mandext_methods[] = {
    {"foo_point", foo_point, METH_VARARGS, ""},
    {"foo_array", foo_array, METH_VARARGS, ""},
    {NULL, NULL}
};

