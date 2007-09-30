""" Time utilities for Mand.
"""

import time

def duration(s):
    """ Make a nice string representation of a number of seconds.
    """
    w = d = h = m = 0
    if s >= 60:
        m, s = divmod(s, 60)
    if m >= 60:
        h, m = divmod(m, 60)
    if h >= 24:
        d, h = divmod(h, 24)
    if d >= 7:
        w, d = divmod(d, 7)
    dur = []
    if w:
        dur.append("%dw" % w)
    if d:
        dur.append("%dd" % d)
    if h:
        dur.append("%dh" % h)
    if m:
        dur.append("%dm" % m)
    if s:
        if int(s) == s:
            dur.append("%ds" % s)
        else:
            dur.append("%.2fs" % s)
    if dur:
        return " ".join(dur)
    else:
        return "0s"

def future(s):
    """ Make a nice string representation of a point in time in the future,
        s seconds from now.
    """
    now = time.time()
    nowyr, nowmon, nowday, nowhr, nowmin, nowsec, _, _, _ = time.localtime(now)
    thenyr, thenmon, thenday, thenhr, thenmin, thensec, _, _, _ = parts = time.localtime(now + s)
    
    fmt = " "   # An initial space to make the leading-zero thing work right.
    if (nowyr, nowmon, nowday) != (thenyr, thenmon, thenday):
        # A different day: decide how to describe the other day.
        fmt += "%a"
        if s > 6*24*60*60:
            # More than a week away: use a real date.
            fmt += ", %d %b"
            if nowyr != thenyr:
                fmt += " %Y"
        fmt += " "
    # Always show the time
    fmt += "%I:%M:%S%p"
    text = time.strftime(fmt, parts)
    text = text.replace(" 0", " ")      # Trim the leading zeros.
    return text[1:]                     # Trim the initial space.

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        s = eval(sys.argv[1])
        print "%d sec from now is %s" % (s, future(s))
    else:
        for p in range(3,8):
            for m in [1,2,5]:
                s = m*10**p
                print "%10d sec from now is %s" % (s, future(s))
