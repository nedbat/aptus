// The Aptus Engine C extension for computing Mandelbrot fractals (hopefully quickly).

#include "Python.h"
#include "numpy/arrayobject.h"

//#define BIFLOAT 1

typedef double float_t;

typedef struct {
    float_t i, r;
} complex_t;

#ifdef BIFLOAT
typedef struct {
    float_t p, d;
} bifloat_t;

typedef struct {
    bifloat_t i, r;
} bicomplex_t;

#define BIINIP(r,s) ((r) + (s))
#define BIINID(r,s) ((s) - (BIINIP(r,s) - (r)))

#define BIMULP(a,b) ((a).p * (b).p)
#define BIMULD(a,b) (a.p * b.d + a.d * b.p + a.d * b.d)

#define BIADDP(a,b) (a.p + b.p + a.d + b.d)
#define BIADDD(a,b) (a.d + b.d - (BIADDP(a,b) - (a.p + b.p)))

#define BISUBP(a,b) (a.p - b.p + a.d - b.d)
#define BISUBD(a,b) (a.d - b.d - (BISUBP(a,b) - (a.p - b.p)))

#define BINORMP(a) (a.p + a.d)
#define BINORMD(a) (a.d - (BINORMP(a) - a.p))

#endif

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

static float_t epsilon;

#ifdef BIFLOAT
inline int
fequal(bifloat_t a, bifloat_t b)
{
    return (a.p == b.p);
}
#else
inline int
fequal(float_t a, float_t b)
{
    return fabs(a - b) < epsilon;
}
#endif

static PyObject *
get_coords(PyObject *self, PyObject *args)
{
    int xi, yi;
    
    if (!PyArg_ParseTuple(args, "ii", &xi, &yi)) {
        return NULL;
    }

    complex_t c;
    c.r = xy0.r + xi*xyd.r;
    c.i = xy0.i + yi*xyd.i;
    
    return Py_BuildValue("dd", (double)c.r, (double)c.i);
}

static void
dump_number(float_t num)
{
    union {
        float_t f;
        char c[sizeof(float_t)];
    } fc;
    fc.f = num;
    int i;
    for (i = 0; i < sizeof(fc.c); i++) {
        printf("%02x", (unsigned char)fc.c[i]);
    }
    printf("\n");
}

static PyObject *
mandelbrot_count(PyObject *self, PyObject *args)
{
    int xi, yi;
    
    if (!PyArg_ParseTuple(args, "ii", &xi, &yi)) {
        return NULL;
    }

    int count = 0;
#ifdef BIFLOAT
    bicomplex_t c;
    c.r.p = BIINIP(xy0.r, xi*xyd.r);
    c.r.d = BIINID(xy0.r, xi*xyd.r);
    c.i.p = BIINIP(xy0.i, yi*xyd.i);
    c.i.d = BIINID(xy0.i, yi*xyd.i);

    bicomplex_t z = {{0,0},{0,0}};
    bicomplex_t znew;
    bicomplex_t z2;

    bicomplex_t cycle_check = z;
#else
    complex_t c;
    c.r = xy0.r + xi*xyd.r;
    c.i = xy0.i + yi*xyd.i;

    //dump_number(c.r);
    
    complex_t z = {0,0};
    complex_t znew;
    complex_t z2;
    
    complex_t cycle_check = z;
#endif

    int cycle_period = INITIAL_CYCLE_PERIOD;
    int cycle_tries = CYCLE_TRIES;
    int cycle_countdown = cycle_period;

    while (count <= max_iter) {

#ifdef BIFLOAT
        z2.r.p = BIMULP(z.r, z.r);
        z2.r.d = BIMULD(z.r, z.r);
        z2.i.p = BIMULP(z.i, z.i);
        z2.i.d = BIMULD(z.i, z.i);
        if (z2.r.p + z2.r.d + z2.i.p + z2.i.d > 4.0) {
            if (count > stats.maxiter) {
                stats.maxiter = count;
            }
            if (stats.miniter == 0 || count < stats.miniter) {
                stats.miniter = count;
            }
            break;
        }
        bifloat_t tmp;
        tmp.p = BISUBP(z2.r, z2.i);
        tmp.d = BISUBD(z2.r, z2.i);
        tmp.p += c.r.p;
        tmp.d += c.r.d;
        
        znew.r.p = BINORMP(tmp);
        znew.r.d = BINORMD(tmp);
        
        znew.i.p = 2 * BIMULP(z.i, z.r) + c.i.p;
        znew.i.d = 2 * BIMULD(z.i, z.r) + c.i.d;
        z = znew;
#else
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
#endif
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
    return Py_BuildValue("ii", sizeof(double), sizeof(float_t));
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

static PyObject *
try_array(PyObject *self, PyObject *args)
{
    PyArrayObject *arr;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arr)) {
        return NULL;
    }
    
    if (arr == NULL) {
        return NULL;
    }
    
    int i, j;
    for (i = 0; i < PyArray_DIM(arr, 0); i++) {
        for (j = 0; j < PyArray_DIM(arr, 1); j++) {
            /* Change the type here depending on your array data type */
            npy_uint32 val = *(npy_uint32 *)PyArray_GETPTR2(arr, i, j);
            printf("[%d,%d] is %d\n", i, j, val);
        }
    }
    
    return Py_BuildValue("i", PyArray_ITEMSIZE(arr));
}

static PyMethodDef
mandext_methods[] = {
    {"mandelbrot_count", mandelbrot_count, METH_VARARGS, "Compute a mandelbrot count for a point"},
    {"set_params", set_params, METH_VARARGS, "Set parameters"},
    {"set_check_cycles", set_check_cycles, METH_VARARGS, "Set more parameters"},
    {"float_sizes", float_sizes, METH_VARARGS, "Get sizes of float types"},
    {"clear_stats", clear_stats, METH_VARARGS, "Clear the statistic counters"},
    {"get_stats", get_stats, METH_VARARGS, "Get the statistics as a dictionary"},
    {"get_coords", get_coords, METH_VARARGS, "xxx"},
    {"try_array", try_array, METH_VARARGS, ""},
    {NULL, NULL}
};

void
initaptus_engine(void)
{
    Py_InitModule("aptus_engine", mandext_methods);
    import_array();
}
