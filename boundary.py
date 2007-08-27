""" A boundary trace function for Mandelbrot computation.
"""

import numpy

def trace_boundary(count_fn, w, h):
    """ Compute counts for pixels, using a boundary trace technique.
        count_fn(x,y) returns the iteration count for a pixel.
        Returns a numpy array w by h with iteration counts for each pixel.
    """
    
    DOWN, LEFT, UP, RIGHT = range(4)
    turn_right = [LEFT, UP, RIGHT, DOWN]
    turn_left = [RIGHT, DOWN, LEFT, UP]
    counts = numpy.zeros((h, w), dtype=numpy.uint32)
    status = numpy.zeros((h, w), dtype=numpy.uint8)
        
    def pix_count(x, y):
        if status[y,x] == 0:
            c = count_fn(x, -y)
            counts[y,x] = c
            status[y,x] = 1
        else:
            c = counts[y,x]
        return c
    
    for xi in xrange(w):
        for yi in xrange(h):
            c = pix_count(xi, yi)
                
            if status[yi,xi] == 1:
                # Start a boundary trace.
                status[yi,xi] == 2
                curdir = DOWN
                curx, cury = xi, yi
                origx, origy = xi, yi
                lastx, lasty = xi, yi
                start = True
                points = []
                
                # Find all the points on the boundary.
                while True:

                    # Eventually, we reach our starting point. Stop.
                    if not start and (curx,cury) == (origx,origy) and curdir == DOWN:
                        break
                    
                    # Move to the next position. If we're off the field, turn left.
                    if curdir == DOWN:
                        if cury >= h-1:
                            curdir = RIGHT
                            continue
                        cury += 1
                    elif curdir == LEFT:
                        if curx <= 0:
                            curdir = DOWN
                            continue
                        curx -= 1
                    elif curdir == UP:
                        if cury <= 0:
                            curdir = LEFT
                            continue
                        cury -= 1
                    elif curdir == RIGHT:
                        if curx >= w-1:
                            curdir = UP
                            continue
                        curx += 1
                    
                    # Get the count of the next position
                    c2 = pix_count(curx, cury)
                        
                    # If the same color, turn right, else turn left.
                    if c2 == c:
                        status[cury,curx] = 2
                        points.append((curx,cury))
                        lastx, lasty = curx, cury
                        curdir = turn_right[curdir]
                    else:
                        curx, cury = lastx, lasty
                        curdir = turn_left[curdir]
                    
                    start = False
                    
                # Now flood fill the region.  The points list has all the boundary
                # points, so we only need to fill left and right from each of those.
                for ptx, pty in points:
                    curx = ptx
                    while True:
                        curx -= 1
                        if curx < 0:
                            break
                        if status[pty,curx] != 0:
                            break
                        counts[pty,curx] = c
                        status[pty,curx] = 2
                    curx = ptx
                    while True:
                        curx += 1
                        if curx > w-1:
                            break
                        if status[pty,curx] != 0:
                            break
                        counts[pty,curx] = c
                        status[pty,curx] = 2
                    
    return counts
