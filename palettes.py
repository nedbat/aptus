""" Palettes for Mand.
"""

import colorsys

def make_step_palette(colors, steps):
    pal = [None]*(len(colors)*steps)
    for i in range(len(pal)):
        color_index = i//steps
        r0, g0, b0 = colors[color_index]
        r1, g1, b1 = colors[(color_index + 1) % len(colors)]
        step = float(i % steps)/steps
        pal[i] = (
            int(r0 + (r1 - r0) * step),
            int(g0 + (g1 - g0) * step),
            int(b0 + (b1 - b0) * step),
            )
    return pal
    
# Colors taken from Xaos, to get the same rendering.
xaos_colors = [
    (0, 0, 0),
    (120, 119, 238),
    (24, 7, 25),
    (197, 66, 28),
    (29, 18, 11),
    (135, 46, 71),
    (24, 27, 13),
    (241, 230, 128),
    (17, 31, 24),
    (240, 162, 139),
    (11, 4, 30),
    (106, 87, 189),
    (29, 21, 14),
    (12, 140, 118),
    (10, 6, 29),
    (50, 144, 77),
    (22, 0, 24),
    (148, 188, 243),
    (4, 32, 7),
    (231, 146, 14),
    (10, 13, 20),
    (184, 147, 68),
    (13, 28, 3),
    (169, 248, 152),
    (4, 0, 34),
    (62, 83, 48),
    (7, 21, 22),
    (152, 97, 184),
    (8, 3, 12),
    (247, 92, 235),
    (31, 32, 16)
]

the_palette = make_step_palette(xaos_colors, 8)

# Make an HSV palette
ncolors = 16
s = .7
v = .9
pal = []
for h in xrange(ncolors):
    pal.append(map(lambda x:int(x*256),colorsys.hsv_to_rgb(h*1.0/ncolors,s,v)))
    #pal.append((64,64,64))

# Swap adjacent colors to mix things up a bit.
if 0:
    for i in range(0, len(the_palette), 2):
        the_palette[i+1], the_palette[i] = the_palette[i], the_palette[i+1]

xthe_palette = [(255,192,192), (255,255,255)]
