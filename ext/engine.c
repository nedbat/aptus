// The Aptus Engine C extension for computing Mandelbrot fractals.
// copyright 2007-2010, Ned Batchelder
// http://nedbatchelder.com/code/aptus

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"
#include "structmember.h"

// Type definitions.

// A floating-point number.
typedef double aptfloat;

// A complex number.
typedef struct {
    aptfloat r, i;
} aptcomplex;

// Specific-sized integers.
typedef npy_uint8 u1int;
typedef npy_uint32 u4int;
typedef npy_uint64 u8int;

#define MAX_U4INT NPY_MAX_UINT32

// Macros lifted from Linux kernel.  Use likely(cond) in an if to indicate that
// condition is most likely true, and unlikely(cond) to indicate unlikely to be
// true.  The compiler will arrange the generated code so the straight-line path
// is the likely case.  This helps the CPU pipeline run best.
#ifdef __GNUC__
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)
#else
#define likely(x)       (x)
#define unlikely(x)     (x)
#endif

const u4int MIN_PROGRESS = 1000000;  // Don't call progress unless we've done this many iters.

// Statistics about the computation (not exposed as a Python type).
typedef struct {
    u4int   maxiter;            // Max iteration that isn't in the set.
    u8int   totaliter;          // Total number of iterations.
    u4int   totalcycles;        // Number of cycles detected.
    u4int   minitercycle;       // Min iteration that was a cycle.
    u4int   maxitercycle;       // Max iteration that was finally a cycle.
    u4int   miniter;            // Minimum iteration count.
    u4int   miniteredge;        // Minimum iteration count on the edge.
    u4int   maxedpoints;        // Number of points that exceeded the maxiter.
    u4int   computedpoints;     // Number of points that were actually computed.
    u4int   filledpoints;       // Number of points that were filled.
    u4int   flippedpoints;      // Number of points that were flipped.
    u4int   boundaries;         // Number of boundaries traced.
    u4int   boundariesfilled;   // Number of boundaries filled.
    u4int   longestboundary;    // Most points in a traced boundary.
    u4int   largestfilled;      // Most pixels filled in a boundary.
} ComputeStats;

void
ComputeStats_clear(ComputeStats *stats)
{
    stats->maxiter = 0;
    stats->totaliter = 0;
    stats->totalcycles = 0;
    stats->minitercycle = MAX_U4INT;
    stats->maxitercycle = 0;
    stats->miniter = INT_MAX;
    stats->miniteredge = 0;
    stats->maxedpoints = 0;
    stats->computedpoints = 0;
    stats->filledpoints = 0;
    stats->flippedpoints = 0;
    stats->boundaries = 0;
    stats->boundariesfilled = 0;
    stats->longestboundary = 0;
    stats->largestfilled = 0;
}

// Create a Python dictionary from a ComputeStats

static PyObject *
ComputeStats_AsDict(ComputeStats *stats)
{
    PyObject * statdict = Py_BuildValue(
        "{si,sK,sI,sI,sI,si,si,sI,sI,sI,sI,sI,sI,sI,sI}",
        "maxiter", stats->maxiter,
        "totaliter", stats->totaliter,
        "totalcycles", stats->totalcycles,
        "minitercycle", stats->minitercycle,
        "maxitercycle", stats->maxitercycle,
        "miniter", stats->miniter,
        "miniteredge", stats->miniteredge,
        "maxedpoints", stats->maxedpoints,
        "computedpoints", stats->computedpoints,
        "filledpoints", stats->filledpoints,
        "flippedpoints", stats->flippedpoints,
        "boundaries", stats->boundaries,
        "boundariesfilled", stats->boundariesfilled,
        "longestboundary", stats->longestboundary,
        "largestfilled", stats->largestfilled
        );

    // Clean up sentinel values for stats that could have no actual value.
    if (stats->miniter == INT_MAX) {
        if (PyDict_SetItemString(statdict, "miniter", Py_None) < 0) {
            return NULL;
        }
    }
    if (stats->minitercycle == MAX_U4INT) {
        if (PyDict_SetItemString(statdict, "minitercycle", Py_None) < 0) {
            return NULL;
        }
    }
    if (stats->maxitercycle == 0) {
        if (PyDict_SetItemString(statdict, "maxitercycle", Py_None) < 0) {
            return NULL;
        }
    }

    return statdict;
}

// The Engine type.

typedef struct {
    PyObject_HEAD
    aptcomplex ri0;         // upper-left point (a pair of floats)
    aptcomplex ridx;        // delta per pixel in x direction (a pair of floats)
    aptcomplex ridy;        // delta per pixel in y direction (a pair of floats)
    aptcomplex rijulia;     // julia point.

    u4int iter_limit;       // limit on iteration count.
    int check_cycles;       // should we check for cycles?
    aptfloat epsilon;       // the epsilon to use when checking for cycles.
    int cont_levels;        // the number of continuous levels to compute.
    int blend_colors;       // how many levels of color should we blend?
    int trace_boundary;     // should we use boundary tracing?
    int julia;              // are we doing julia or mandelbrot?

    // Parameters controlling the cycle detection.
    struct {
        int     initial_period; // save every nth z value as a cycle check.
        int     tries;          // use each period value this many times before choosing a new period.
        int     factor;         // to get a new period, multiply by this,
        int     delta;          //  .. and add this.
    } cycle_params;

    PyObject * debug_callback;  // Maybe defined.
} AptEngine;


// Class methods

static int
AptEngine_init(AptEngine *self, PyObject *args, PyObject *kwds)
{
    self->ri0.r = 0.0;
    self->ri0.i = 0.0;
    self->ridx.r = 0.001;
    self->ridx.i = 0.0;
    self->ridy.r = 0.0;
    self->ridy.i = 0.001;
    self->rijulia.r = 0.0;
    self->rijulia.i = 0.0;
    self->iter_limit = 1000;
    self->check_cycles = 1;
    self->trace_boundary = 1;
    self->cont_levels = 1;
    self->blend_colors = 1;
    self->julia = 0;

    self->cycle_params.initial_period = 43;
    self->cycle_params.tries = 10;
    self->cycle_params.factor = 2;
    self->cycle_params.delta = -1;

    self->debug_callback = Py_BuildValue("");

    return 0;
}

static void
AptEngine_dealloc(AptEngine *self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// ri0 property methods

static PyObject *
get_ri0(AptEngine *self, void *closure)
{
    return Py_BuildValue("dd", self->ri0.r, self->ri0.i);
}

static int
set_ri0(AptEngine *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the ri0 attribute");
        return -1;
    }

    if (!PyArg_ParseTuple(value, "dd", &self->ri0.r, &self->ri0.i)) {
        return -1;
    }

    return 0;
}

// ridxdy property methods

static PyObject *
get_ridxdy(AptEngine *self, void *closure)
{
    return Py_BuildValue("dddd", self->ridx.r, self->ridx.i, self->ridy.r, self->ridy.i);
}

static int
set_ridxdy(AptEngine *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the ridxdy attribute");
        return -1;
    }

    if (!PyArg_ParseTuple(value, "dddd", &self->ridx.r, &self->ridx.i, &self->ridy.r, &self->ridy.i)) {
        return -1;
    }

    // Make a crude estimate of an epsilon to use for cycle checking.
    self->epsilon = (self->ridx.r+self->ridx.i)/2;

    return 0;
}

// rijulia property methods

static PyObject *
get_rijulia(AptEngine *self, void *closure)
{
    return Py_BuildValue("dd", self->rijulia.r, self->rijulia.i);
}

static int
set_rijulia(AptEngine *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the rijulia attribute");
        return -1;
    }

    if (!PyArg_ParseTuple(value, "dd", &self->rijulia.r, &self->rijulia.i)) {
        return -1;
    }

    return 0;
}

// cycle_params property methods

static PyObject *
get_cycle_params(AptEngine *self, void *closure)
{
    return Py_BuildValue("iiii",
        self->cycle_params.initial_period,
        self->cycle_params.tries,
        self->cycle_params.factor,
        self->cycle_params.delta
        );
}

static int
set_cycle_params(AptEngine *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the cycle_params attribute");
        return -1;
    }

    if (!PyArg_ParseTuple(value, "iiii",
        &self->cycle_params.initial_period,
        &self->cycle_params.tries,
        &self->cycle_params.factor,
        &self->cycle_params.delta
        ))
    {
        return -1;
    }

    return 0;
}

// Are two floating point numbers equal?
inline int
fequal(AptEngine *self, aptfloat a, aptfloat b)
{
    return fabs(a - b) < self->epsilon;
}

// An experiment in leaving out statistics and progress reporting.
// No idea why, but making these empty macros (thereby removing a
// bunch of code) makes the code *slower*.
#define STATS_DECL(x)   x
#define STATS_CODE(x)   x

// compute_count: the heart of the Mandelbrot algorithm.
//
// Given an integer coordinate xi,yi, return the iteration count for that point
// in the current array.  If continuous coloring, then the returned value is
// scaled up (a fixed-point number essentially).

static u4int
compute_count(AptEngine *self, int xi, int yi, ComputeStats *stats)
{
    u4int count = 0;

    // The complex point we're computing for.
    aptcomplex c;

    // z is the value we're iterating, znew and z2 are intermediates.
    aptcomplex z;
    aptcomplex znew;
    aptcomplex z2;

    if (self->julia) {
        c.r = self->rijulia.r;
        c.i = self->rijulia.i;
        z.r = self->ri0.r + xi*self->ridx.r + yi*self->ridy.r;
        z.i = self->ri0.i + xi*self->ridx.i + yi*self->ridy.i;
    }
    else {
        c.r = self->ri0.r + xi*self->ridx.r + yi*self->ridy.r;
        c.i = self->ri0.i + xi*self->ridx.i + yi*self->ridy.i;
        z.r = 0.0;
        z.i = 0.0;
    }

    // Cycle checking bookkeeping variables.
    aptcomplex cycle_check = z;

    int cycle_period = self->cycle_params.initial_period;
    int cycle_tries = self->cycle_params.tries;
    int cycle_countdown = cycle_period;

    aptfloat bailout = self->cont_levels > 1 ? 100.0 : 2.0;
    aptfloat bail2 = bailout * bailout;

    // Macros for the meat of the iterations, since we need to do them in a few
    // places.
#define ITER1                               \
    z2.r = z.r * z.r;                       \
    z2.i = z.i * z.i;

#define ITER2                               \
    znew.r = z2.r - z2.i + c.r;             \
    znew.i = 2 * z.i * z.r + c.i;           \
    z = znew;                               \
    STATS_CODE(stats->totaliter++;)

    // First phase of iterations: quickly iterate up to the minimum iteration
    // count.
    register int miniter = stats->miniteredge;
    while (likely(miniter-- > 0)) {
        ITER1; ITER2;
    }
    count = stats->miniteredge;

    // Second phase: iterate more carefully, looking for bailout, cycles, etc.
    while (likely(count <= self->iter_limit)) {
        ITER1;
        if (unlikely(z2.r + z2.i > bail2)) {
            // The point has escaped the bailout.  Update the stats and bail out.
            STATS_CODE(
            if (unlikely(count > stats->maxiter)) {
                stats->maxiter = count;
            }
            if (unlikely(count < stats->miniter)) {
                stats->miniter = count;
            }
            )
            break;
        }
        ITER2;

        if (likely(self->check_cycles)) {
            // Check for cycles
            if (unlikely(fequal(self, z.r, cycle_check.r) && fequal(self, z.i, cycle_check.i))) {
                // We're in a cycle! Update stats, and end the iterations.
                STATS_CODE(
                stats->totalcycles++;
                if (unlikely(count > stats->maxitercycle)) {
                    stats->maxitercycle = count;
                }
                if (unlikely(count < stats->minitercycle)) {
                    stats->minitercycle = count;
                }
                )
                // A cycle means we're inside the set (count of 0).
                count = 0;
                break;
            }

            if (--cycle_countdown == 0) {
                // Take a new cycle_check point.
                cycle_check = z;
                cycle_countdown = cycle_period;
                if (--cycle_tries == 0) {
                    cycle_period = self->cycle_params.factor*cycle_period-self->cycle_params.delta;
                    cycle_tries = self->cycle_params.tries;
                }
            }
        }

        count++;
    }

    // Counts above the iteration limit are colored as if they were in the set.
    if (unlikely(count > self->iter_limit)) {
        STATS_CODE(stats->maxedpoints++;)
        count = 0;
    }

    // Smooth coloring.
    if (count > 0 && self->cont_levels != 1) {
        // http://linas.org/art-gallery/escape/smooth.html

        // three more iterations to reduce the error.
        ITER2;      // we didn't finish the last one.
        ITER1; ITER2;
        ITER1; ITER2;
        ITER1; ITER2;

        // The 2 here is the power of the iteration, not the bailout.
        const double log2 = 0.69314718055994529;
        double delta = log(log(sqrt(z2.r + z2.i)))/log2;
        double fcount = count + 3 - delta;
        if (unlikely(fcount < 1)) {
            // Way outside the set, continuous mode acts weird.  Cut it off at 1.
            fcount = 1;
        }
        count = (u4int)(fcount * self->cont_levels);
    }

    STATS_CODE(stats->computedpoints++;)

    return count;
}

// Helper: call_progress

STATS_DECL(
static int
call_progress(AptEngine *self, PyObject *progress, PyObject *prog_arg, u4int num_complete, char *info)
{
    int ok = 1;
    PyObject * result = PyObject_CallFunction(progress, "OIs", prog_arg, num_complete, info);
    if (result == NULL) {
        ok = 0;
    }
    Py_XDECREF(result);
    return ok;
}
)

// A debug callback, usually not compiled in.

#if 0
#define CALL_DEBUG(info)                            \
        Py_BLOCK_THREADS                            \
        ret = call_debug(self, info);               \
        Py_UNBLOCK_THREADS                          \
        if (!ret) {                                 \
            goto done;                              \
        }

static int
call_debug(AptEngine *self, char *info)
{
    int ok = 1;
    if (self->debug_callback != Py_None) {
        PyObject * result = PyObject_CallFunction(self->debug_callback, "s", info);
        if (result == NULL) {
            ok = 0;
        }
        Py_XDECREF(result);
    }
    return ok;
}

#else
#define CALL_DEBUG(info)
#endif

// Helper: display a really big number in a portable way

static char *
human_u8int(u8int big, char *buf)
{
    float little = (float)big;
    if (big < 10000000) {   // 10 million
        sprintf(buf, "%u", (u4int)big);
    }
    else if (big < 1000000000) {    // 1 billion
        little /= 1e6;
        sprintf(buf, "%.1fM", little);
    }
    else {
        little /= 1e9;
        sprintf(buf, "%.1fB", little);
    }

    return buf;
}

// compute_array

static char compute_array_doc[] = "Compute Mandelbrot counts for an array";

static PyObject *
compute_array(AptEngine *self, PyObject *args)
{
    // Arguments to compute_array
    // counts is an array of iteration counts (the actual output of the function).
    PyArrayObject *counts;
    // status is an array of the status of the pixels.
    //  0: hasn't been computed yet.
    //  1: computed, but not filled.
    //  2: computed and filled.
    PyArrayObject *status;
    // The corners of the rectangle to compute.
    int xmin, xmax, ymin, ymax;
    // An opaque argument to pass to the progress function.
    PyObject * prog_arg;
    // `progress` is a Python callable, the progress reporting callback. It
    // takes three arguments: the opaque argument, the number of pixels
    // finished, and a string of information.
    PyObject * progress;

    int ok = 0;
    int ret;

    // Malloc'ed buffers.
    // points is an array of points on a boundary.
    typedef struct { int x, y; } Point;
    Point * points = NULL;

    if (!PyArg_ParseTuple(args, "O!O!iiiiOO:compute_array",
            &PyArray_Type, &counts, &PyArray_Type, &status,
            &xmin, &xmax, &ymin, &ymax,
            &prog_arg, &progress)) {
        goto early_done;
    }

    if (!PyCallable_Check(progress)) {
        PyErr_SetString(PyExc_TypeError, "progress must be callable");
        goto early_done;
    }

    u4int ptsalloced = 10;
    u4int ptsstored = 0;
    points = PyMem_New(Point, ptsalloced);

    ComputeStats stats;
    ComputeStats_clear(&stats);

    u4int num_pixels = 0;

    Py_BEGIN_ALLOW_THREADS

    STATS_DECL(
    // Progress reporting stuff.
    char info[100];
    char uinfo[100];
    u8int last_progress;    // the totaliter the last time we called the progress function.
    last_progress = 0;
    )

// Convenient accessors for our arrays
#define STATUS(x,y) *(npy_uint8 *)PyArray_GETPTR2(status, (y), (x))
#define COUNTS(x,y) *(npy_uint32 *)PyArray_GETPTR2(counts, (y), (x))

    enum {
        DIR_DOWN, DIR_LEFT, DIR_UP, DIR_RIGHT
    };

    enum {
        STATUS_UNCOMPUTED, STATUS_UNTRACED, STATUS_TRACING, STATUS_FILLED
    };

    int xi, yi;
    u1int s;
    int c;
    u4int pi;
    int ptx, pty;

    // Figure out if we can flip around the x-axis.
    int flipping = 0;
    aptfloat axisy = 0;
    int fliplo = 0, fliphi = 0;

    if (!self->julia && self->ridx.r != 0 && self->ridx.i == 0 && self->ridy.r == 0 && self->ridy.i != 0) {
        // The symmetry axis is horizontal.
        axisy = self->ri0.i/-self->ridy.i;
        aptfloat above = self->ri0.i + floor(axisy)*self->ridy.i;
        aptfloat below = self->ri0.i + ceil(axisy)*self->ridy.i;

        // printf("above = %f, below = %f\n", above, below);
        if (fequal(self, above, -below)) {
            // printf("Properly aligned!\n");
            if (ymin < axisy && axisy < ymax-1) {
                // printf("Flipping!\n");
                flipping = 1;
                fliphi = (int)floor(axisy);
                fliplo = fliphi - (ymax - fliphi);
            }
        }
    }

    int flipped = 0;
    int yflip = 0;

// A macro to potentially flip a result across the x axis.
// TODO: This macro checks the status of the flipped point to see that it is
// UNCOMPUTED.  But I don't see why that's necessary: if the flipped point is
// some other state, then why wasn't it flipped up to the original point?  It's
// a pair of points that should move in lockstep.  For now, at least check that
// status so we don't duplicate work.
#define FLIP_POINT(xi, yi, c, s)                            \
        if (flipping && fliplo < yi && yi < fliphi) {       \
            int yother = (int)(axisy + (axisy - yi));       \
            if (STATUS(xi, yother) == STATUS_UNCOMPUTED) {  \
                COUNTS(xi, yother) = c;                     \
                STATUS(xi, yother) = s;                     \
                stats.flippedpoints++;                      \
                num_pixels++;                               \
                flipped = 1;                                \
                yflip = yother;                             \
            }                                               \
        }

// A macro to get s and c for a particular point.
#define CALC_POINT(xi, yi)                              \
        s = STATUS(xi, yi);                             \
        if (s == STATUS_UNCOMPUTED) {                   \
            c = compute_count(self, xi, yi, &stats);    \
            COUNTS(xi, yi) = c;                         \
            num_pixels++;                               \
            STATUS(xi, yi) = s = STATUS_UNTRACED;       \
            FLIP_POINT(xi, yi, c, STATUS_UNTRACED);     \
            CALL_DEBUG("point");                        \
        }                                               \
        else {                                          \
            c = COUNTS(xi, yi);                         \
        }

    // Walk the edges of the array to find the minimum iteration count.  If
    // boundary tracing is allowed, then the minimum iteration count is guaranteed
    // to exist along one of the edges.  We can quickly find the minimum, and then
    // use that value in compute_count to quickly iterate to the minimum without
    // checking the bailout condition.
    int miniteredge = INT_MAX;
    if (self->trace_boundary) {
        // Calc the left and right edges
        for (yi = ymin; yi < ymax; yi++) {
            CALC_POINT(xmin, yi);
            if (c < miniteredge) {
                miniteredge = c;
            }
            CALC_POINT(xmax-1, yi);
            if (c < miniteredge) {
                miniteredge = c;
            }
        }
        // Calc the top and bottom edges
        for (xi = xmin+1; xi < xmax-1; xi++) {
            CALC_POINT(xi, ymin);
            if (c < miniteredge) {
                miniteredge = c;
            }
            CALC_POINT(xi, ymax-1);
            if (c < miniteredge) {
                miniteredge = c;
            }
        }
        stats.miniteredge = (u4int)(miniteredge / self->cont_levels);
    }
    else {
        stats.miniteredge = 0;
    }

    // Loop the pixels in the array.
    for (yi = ymin; yi < ymax; yi++) {
        for (xi = xmin; xi < xmax; xi++) {
            // Examine the current pixel.
            s = STATUS(xi, yi);
            if (s == STATUS_UNCOMPUTED) {
                c = compute_count(self, xi, yi, &stats);
                COUNTS(xi, yi) = c;
                num_pixels++;
                STATUS(xi, yi) = s = STATUS_UNTRACED;
                FLIP_POINT(xi, yi, c, STATUS_UNTRACED);
                CALL_DEBUG("point");
            }
            else if (s == STATUS_UNTRACED && self->trace_boundary) {
                c = COUNTS(xi, yi);
            }

            // A pixel that's been calculated but not traced needs to be traced.
            if (s == STATUS_UNTRACED && self->trace_boundary) {
                const char FIRST_DIR = DIR_UP;
                char curdir = FIRST_DIR;
                int curx = xi, cury = yi;
                int origx = xi, origy = yi;
                int lastx = xi, lasty = yi;
                int start = 1;

                STATUS(xi, yi) = STATUS_TRACING;

                points[0].x = curx;
                points[0].y = cury;
                ptsstored = 1;

                // Walk the boundary
                for (;;) {
                    // Eventually, we reach our starting point. Stop.
                    if (unlikely(!start && curx == origx && cury == origy && curdir == FIRST_DIR)) {
                        break;
                    }

                    // Move to the next position. If we're off the field, turn
                    // left (same as if the pixel off-field were a different color
                    // than us).
                    switch (curdir) {
                    case DIR_DOWN:
                        if (unlikely(cury >= ymax-1)) {
                            curdir = DIR_RIGHT;
                            continue;
                        }
                        cury++;
                        break;

                    case DIR_LEFT:
                        if (unlikely(curx <= xmin)) {
                            curdir = DIR_DOWN;
                            continue;
                        }
                        curx--;
                        break;

                    case DIR_UP:
                        if (unlikely(cury <= ymin)) {
                            curdir = DIR_LEFT;
                            continue;
                        }
                        cury--;
                        break;

                    case DIR_RIGHT:
                        if (unlikely(curx >= xmax-1)) {
                            curdir = DIR_UP;
                            continue;
                        }
                        curx++;
                        break;
                    }

                    // Get the count of the next position.
                    int c2 = 0;
                    flipped = 0;
                    s = STATUS(curx, cury);
                    switch (s) {
                    case STATUS_UNCOMPUTED:
                        c2 = compute_count(self, curx, cury, &stats);
                        COUNTS(curx, cury) = c2;
                        num_pixels++;
                        STATUS(curx, cury) = STATUS_UNTRACED;
                        FLIP_POINT(curx, cury, c2, STATUS_UNTRACED);
                        CALL_DEBUG("point");
                        break;

                    case STATUS_UNTRACED:
                        c2 = COUNTS(curx, cury);
                        break;

                    case STATUS_TRACING:
                        c2 = c; // Should be...
                        break;

                    case STATUS_FILLED:
                        // Don't wander into filled territory.
                        c2 = c+1;
                        break;
                    }

                    // If the same color, turn right, else turn left.
                    if (c2 == c) {
                        STATUS(curx, cury) = STATUS_TRACING;
                        if (flipped) {
                            STATUS(curx, yflip) = STATUS_TRACING;
                        }
                        // Append the point to the points list, growing dynamically
                        // if we have to.
                        if (unlikely(ptsstored == ptsalloced)) {
                            Point * newpts = points;
                            Py_BLOCK_THREADS
                            PyMem_Resize(newpts, Point, ptsalloced*2);
                            Py_UNBLOCK_THREADS
                            if (newpts == NULL) {
                                Py_BLOCK_THREADS
                                PyErr_SetString(PyExc_MemoryError, "couldn't allocate points");
                                Py_UNBLOCK_THREADS
                                goto late_done;
                            }
                            points = newpts;
                            ptsalloced *= 2;
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

                STATS_CODE(
                stats.boundaries++;
                if (ptsstored > stats.longestboundary) {
                    stats.longestboundary = ptsstored;
                }
                )

                // If we saved enough boundary points, then we flood fill. The
                // points are orthogonally connected, so we need at least eight
                // to enclose a fillable point.
                if (unlikely(ptsstored >= 8)) {
                    // Flood fill the region. The points list has all the boundary
                    // points, so we only need to fill left from each of those.
                    // This works because the boundary points are ringed by all
                    // the points with the wrong color, so there's no chance that
                    // filling left from the left edge will flood outside the
                    // region.
                    u4int num_filled = 0;
                    for (pi = 0; pi < ptsstored; pi++) {
                        ptx = points[pi].x;
                        pty = points[pi].y;
                        STATUS(ptx, pty) = STATUS_FILLED;
                        // Fill left from points where we moved up.
                        for (;;) {
                            ptx--;
                            if (ptx < xmin) {
                                break;
                            }
                            if (STATUS(ptx, pty) != STATUS_UNCOMPUTED) {
                                break;
                            }
                            COUNTS(ptx, pty) = c;
                            num_pixels++;
                            num_filled++;
                            STATUS(ptx, pty) = STATUS_FILLED;
                            CALL_DEBUG("fill");
                        }
                    } // end for points to fill

                    STATS_CODE(
                    if (num_filled > 0) {
                        stats.filledpoints += num_filled;
                        stats.boundariesfilled++;

                        // If this was a large boundary, call the progress function.
                        if (ptsstored > (u4int)(xmax-xmin)) {
                            if ((stats.totaliter - last_progress) > MIN_PROGRESS) {
                                sprintf(info, "trace %d * %d, totaliter %s", c, ptsstored, human_u8int(stats.totaliter, uinfo));
                                Py_BLOCK_THREADS
                                ret = call_progress(self, progress, prog_arg, num_pixels, info);
                                Py_UNBLOCK_THREADS
                                if (!ret) {
                                    goto late_done;
                                }
                                last_progress = stats.totaliter;
                            }
                        }

                        if (num_filled > stats.largestfilled) {
                            stats.largestfilled = num_filled;
                        }
                    }
                    )
                } // end if points
            } // end if needs trace
        } // end for xi

        STATS_CODE(
        // At the end of the scan line, call progress if we've made enough progress
        if (stats.totaliter - last_progress > MIN_PROGRESS) {
            sprintf(info, "scan %d, totaliter %s", yi+1, human_u8int(stats.totaliter, uinfo));
            Py_BLOCK_THREADS
            ret = call_progress(self, progress, prog_arg, num_pixels, info);
            Py_UNBLOCK_THREADS
            if (!ret) {
                goto late_done;
            }
            last_progress = stats.totaliter;
        }
        )
    } // end for yi

    // Clean up.
    ok = 1;

late_done:
    Py_END_ALLOW_THREADS

early_done:
    // Free allocated memory.
    if (points != NULL) {
        PyMem_Free(points);
    }

    return ok ? ComputeStats_AsDict(&stats) : NULL;
}

// apply_palette

static char apply_palette_doc[] = "Color an array based on counts and palette";

static PyObject *
apply_palette(AptEngine *self, PyObject *args)
{
    // Arguments to the function.
    PyArrayObject *counts;
    PyObject * status_obj;
    PyArrayObject *status;
    PyObject * colbytes_obj;
    PyObject * incolor_obj;
    PyArrayObject *pix;
    int phase;
    double scale;
    int wrap;

    // Objects we get during the function.
    PyObject * pint = NULL;
    int ok = 0;

    if (!PyArg_ParseTuple(args, "O!OOidOiO!:apply_palette", &PyArray_Type, &counts, &status_obj, &colbytes_obj, &phase, &scale, &incolor_obj, &wrap, &PyArray_Type, &pix)) {
        goto done;
    }

    if (status_obj != Py_None) {
        status = (PyArrayObject*)status_obj;
    }
    else {
        status = NULL;
    }

    // Unpack the palette a bit.
    u1int * colbytes;
    Py_ssize_t ncolbytes;
    if (PyBytes_AsStringAndSize(colbytes_obj, (char**)&colbytes, &ncolbytes) < 0) {
        goto done;
    }
    int ncolors = ncolbytes / 3;

    u1int incolbytes[3];
    int i;
    for (i = 0; i < 3; i++) {
        pint = PySequence_GetItem(incolor_obj, i);
        incolbytes[i] = (u1int)PyLong_AsLong(pint);
        Py_CLEAR(pint);
    }

    // A one-element cache of count and color.
    npy_uint8 *plastpix = NULL;
    npy_uint32 lastc = (*(npy_uint32 *)PyArray_GETPTR2(counts, 0, 0))+1;    // Something different than the first value.

// A macro to deal with out-of-range color indexes
#define WRAP_COLOR(cindex)                      \
        if (wrap) {                             \
            cindex %= ncolors;                  \
        }                                       \
        else {                                  \
            if (cindex < 0) {                   \
                cindex = 0;                     \
            }                                   \
            else if (cindex >= ncolors) {       \
                cindex = ncolors - 1;           \
            }                                   \
        }

    // Walk the arrays
    int h = PyArray_DIM(counts, 0);
    int w = PyArray_DIM(counts, 1);
    int count_stride = PyArray_STRIDE(counts, 1);
    int pix_stride = PyArray_STRIDE(pix, 1);

    int x, y;
    for (y = 0; y < h; y++) {
        // The count for this pixel.
        char *pcount = (char*)PyArray_GETPTR2(counts, y, 0);
        // The pointer to the pixel's RGB bytes.
        npy_uint8 *ppix = (npy_uint8 *)PyArray_GETPTR3(pix, y, 0, 0);

        for (x = 0; x < w; x++) {
            // Don't touch pixels that haven't been computed.
            if (!status || STATUS(x, y) != 0) {
                npy_uint32 c = *(npy_uint32*)pcount;
                if (c == lastc) {
                    // A hit on the one-element cache. Copy the old value.
                    memcpy(ppix, plastpix, 3);
                }
                else {
                    if (c > 0) {
                        // The pixel is outside the set, color it with the palette.
                        if (self->blend_colors == 1) {
                            // Not blending colors, each count is a literal palette
                            // index.
                            int cindex = c + phase;
                            WRAP_COLOR(cindex)
                            memcpy(ppix, (colbytes + cindex*3), 3);
                        }
                        else {
                            double cf = c * scale / self->blend_colors;
                            int cbase = (int)cf;
                            float cfrac = (float)(cf - cbase);
                            int c1index = cbase + phase;
                            int c2index = cbase + 1 + phase;
                            WRAP_COLOR(c1index)
                            WRAP_COLOR(c2index)
                            for (i = 0; i < 3; i++) {
                                float col1 = colbytes[c1index*3+i];
                                float col2 = colbytes[c2index*3+i];
                                ppix[i] = (int)(col1 + (col2-col1)*cfrac);
                            }
                        }
                    }
                    else {
                        // The pixel is in the set, color it with the incolor.
                        memcpy(ppix, incolbytes, 3);
                    }

                    // Save this value in our one-element cache.
                    lastc = c;
                    plastpix = ppix;
                }
            }

            // Advance to the next pixel.
            pcount += count_stride;
            ppix += pix_stride;
        }
    }

    ok = 1;

done:
    Py_XDECREF(pint);

    return ok ? Py_BuildValue("") : NULL;
}

// type_check

static char type_check_doc[] = "Try out types in the C extension";

static PyObject *
type_check(PyObject *self, PyObject *args)
{
    char info[200];
    char uinfo[200];
    u8int big = 1;
    big <<= 40;
    sprintf(info, "Big 1<<40 = %s", human_u8int(big, uinfo));

    return Py_BuildValue(
        "{si,si,si,si,ss}",
        "double", sizeof(double),
        "aptfloat", sizeof(aptfloat),
        "u4int", sizeof(u4int),
        "u8int", sizeof(u8int),
        "sprintf", info
        );
}

// Type definition

static PyMemberDef
AptEngine_members[] = {
    { "iter_limit",     T_INT,      offsetof(AptEngine, iter_limit),        0, "Limit on iterations" },
    { "check_cycles",   T_INT,      offsetof(AptEngine, check_cycles),      0, "Check for cycles?" },
    { "cont_levels",    T_INT,      offsetof(AptEngine, cont_levels),       0, "Number of fractional levels to compute" },
    { "blend_colors",   T_INT,      offsetof(AptEngine, blend_colors),      0, "How many levels of color to blend" },
    { "trace_boundary", T_INT,      offsetof(AptEngine, trace_boundary),    0, "Control whether boundaries are traced" },
    { "julia",          T_INT,      offsetof(AptEngine, julia),             0, "Compute Julia set?" },
    { "debug_callback", T_OBJECT,   offsetof(AptEngine, debug_callback),    0, "Called to debug" },
    { NULL }
};

static PyGetSetDef
AptEngine_getsetters[] = {
    { "ri0",            (getter)get_ri0,            (setter)set_ri0,            "Upper-left corner coordinates", NULL },
    { "ridxdy",         (getter)get_ridxdy,         (setter)set_ridxdy,         "Pixel offsets", NULL },
    { "rijulia",        (getter)get_rijulia,        (setter)set_rijulia,        "Julia point", NULL },
    { "cycle_params",   (getter)get_cycle_params,   (setter)set_cycle_params,   "Cycle detection parameters", NULL },
    { NULL }
};

static PyMethodDef
AptEngine_methods[] = {
    { "compute_array",      (PyCFunction) compute_array,    METH_VARARGS, compute_array_doc },
    { "apply_palette",      (PyCFunction) apply_palette,    METH_VARARGS, apply_palette_doc },
    { NULL }
};

static PyTypeObject
AptEngineType = {
    PyVarObject_HEAD_INIT(NULL, 0)
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
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
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
    0,                         /* tp_new */
};


// Module definition

#define MODULE_DOC PyDoc_STR("Fast Aptus Mandelbrot engine.")

static PyMethodDef
AptEngine_functions[] = {
    { "type_check", type_check, METH_VARARGS, type_check_doc },
    { NULL }
};

static PyModuleDef
moduledef = {
    PyModuleDef_HEAD_INIT,
    "aptus.engine",
    MODULE_DOC,
    -1,
    AptEngine_functions,    /* methods */
    NULL,                   /* slots */
    NULL,                   /* traverse */
    NULL,                   /* clear */
    NULL                    /* free */
};

PyObject *
PyInit_engine(void)
{
    import_array();

    PyObject * mod = PyModule_Create(&moduledef);
    if (mod == NULL) {
        return NULL;
    }

    AptEngineType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&AptEngineType) < 0) {
        Py_DECREF(mod);
        return NULL;
    }

    Py_INCREF(&AptEngineType);
    if (PyModule_AddObject(mod, "AptEngine", (PyObject *)&AptEngineType) < 0) {
        Py_DECREF(mod);
        Py_DECREF(&AptEngineType);
        return NULL;
    }

    return mod;
}
