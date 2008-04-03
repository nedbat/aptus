# Pure python (slow, out-of-date, and incomplete!) implementation of aptus_engine interface.

class AptEngine:
    MAXITER = 999
    X0, Y0 = 0, 0
    XD, YD = 0, 0
    
    def mandelbrot_count(self, xi, yi):
        p = complex(X0+xi*XD, Y0+yi*YD)
        i = 0
        z = 0+0j
        while abs(z) < 2:
            if i >= MAXITER:
                return 0
            z = z*z+p
            i += 1
        return i

    def set_geometry(self, x0, y0, xd, yd):
        global X0, Y0, XD, YD
        X0, Y0 = x0, y0
        XD, YD = xd, yd

    def set_maxiter(self, maxiter):
        global MAXITER
        MAXITER = maxiter

    def clear_stats(self):
        pass

    def get_stats(self):
        return {}
