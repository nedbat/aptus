#----------------------------------------------------------------------
# This file was generated by /Python25/Scripts/img2py
#
from wx import Image, BitmapFromImage
import cStringIO, zlib


def getCrosshairData():
    return zlib.decompress(
"x\xda\xeb\x0c\xf0s\xe7\xe5\x92\xe2b``\xe0\xf5\xf4p\t\x02\xd2\x02 \xcc\xc1\
\x06$\xe5?\xffO\x04R,\xc5N\x9e!\x1c@P\xc3\x91\xd2\x01\xe4g{\xba8\x86X\xf4\
\xde\x9dt\x90\xeb\x80\x01\x87s\xed\xad\xff\xff\x1f\xeb\xed\xde\xb2xC\xd8)=\
\xe9\xd3\x9e\x01\xc9\x1e,\xfc\xd1\x8e\xee\xa2r:\xecg\xcelN\xf5\\\xb6\xf8\x95\
\xe0A\xbd:\x07\x06\xb9\xe0\xc5n\xdf7\xd7\xbd\xcc\xbb\xbf\xc4u\xe7\xabHy\x019\
\x11\x1b\x864\xf1c\x0f\xad\x1d\x83\x0f\xbc\x9a\xb8\xcb\xe8\xc0\xa6\xf4*\xfbV\
~\xe1-%A\xccg'\x89\x94\x1525.\xd2\ngpRI\x03Z\xcb\xe0\xe9\xea\xe7\xb2\xce)\
\xa1\t\x00)E<\xd9" )

def getCrosshairBitmap():
    return BitmapFromImage(getCrosshairImage())

def getCrosshairImage():
    stream = cStringIO.StringIO(getCrosshairData())
    return Image(stream)
