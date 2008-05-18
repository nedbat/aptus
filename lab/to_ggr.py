from palettes import xaos_colors
#xaos_colors = xaos_colors[:10]
print "GIMP Gradient"
print "Name: Xaos"
print "%d" % len(xaos_colors)
fmt = "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %d %d"
for i in range(len(xaos_colors)):
    i1 = (i+1) % len(xaos_colors)
    cl = xaos_colors[i]
    cr = xaos_colors[i1]
    sl = float(i)/len(xaos_colors)
    sr = float(i1)/len(xaos_colors)
    if sr < sl:
        assert sr == 0
        sr = 1
    print fmt % (sl, (sl+sr)/2, sr, cl[0]/255.0, cl[1]/255.0, cl[2]/255.0, 1, cr[0]/255.0, cr[1]/255.0, cr[2]/255.0, 1, 0, 0)
    