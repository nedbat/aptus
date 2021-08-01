import ast
import asyncio
import base64
import dataclasses
import functools
import io
import os
import pathlib

import cachetools
import PIL
import pydantic
import uvicorn

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from aptus import __version__
from aptus.compute import AptusCompute
from aptus.palettes import Palette, all_palettes

app = FastAPI()

HERE = pathlib.Path(__file__).parent
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    context = {
        "request": request,
        "palettes": [p.spec() for p in all_palettes],
        "version": __version__,
    }
    return templates.TemplateResponse("mainpage.html", context)

def run_in_executor(f):
    # from https://stackoverflow.com/a/53719009/14343
    @functools.wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, lambda: f(*args, **kwargs))
    return inner


@dataclasses.dataclass
class CachedResult:
    counts: object  # ndarray
    stats: dict

@dataclasses.dataclass
class TileResult:
    pixels: bytes
    stats: dict


# Cache of computed counts. One tile is about 830Kb.
cache_size = int(os.getenv("APTUS_CACHE", "500"))
tile_cache = cachetools.LRUCache(cache_size * 1_000_000, getsizeof=lambda cr: cr.counts.nbytes)

@run_in_executor
def compute_tile(compute, cachekey):
    old = tile_cache.get(cachekey)
    if old is None:
        compute.compute_array()
        stats = compute.stats
        tile_cache[cachekey] = CachedResult(counts=compute.counts, stats=stats)
    else:
        compute.set_counts(old.counts)
        stats = old.stats
    pix = compute.color_mandel()
    im = PIL.Image.fromarray(pix)
    fout = io.BytesIO()
    compute.write_image(im, fout)
    return TileResult(pixels=fout.getvalue(), stats=stats)

@run_in_executor
def compute_render(compute):
    compute.compute_pixels()
    pix = compute.color_mandel()
    im = PIL.Image.fromarray(pix)
    if compute.supersample > 1:
        im = im.resize(compute.size, PIL.Image.ANTIALIAS)
    fout = io.BytesIO()
    compute.write_image(im, fout)
    return fout.getvalue()

class ComputeSpec(pydantic.BaseModel):
    center: tuple[float, float]
    diam: tuple[float, float]
    size: tuple[int, int]
    supersample: int
    coords: tuple[int, int, int, int]
    angle: float
    continuous: bool
    iter_limit: int
    palette: list
    palette_tweaks: dict

class TileRequest(pydantic.BaseModel):
    spec: ComputeSpec
    seq: int

def spec_to_compute(spec):
    compute = AptusCompute()
    compute.quiet = True
    compute.center = spec.center
    compute.diam = spec.diam
    compute.size = spec.size
    compute.supersample = spec.supersample
    compute.angle = spec.angle
    compute.continuous = spec.continuous
    compute.iter_limit = spec.iter_limit
    compute.palette = Palette().from_spec(spec.palette)
    compute.palette_phase = spec.palette_tweaks.get("phase", 0)
    compute.palette_scale = spec.palette_tweaks.get("scale", 1.0)
    compute.palette.adjust(
        hue=spec.palette_tweaks.get("hue", 0),
        saturation=spec.palette_tweaks.get("saturation", 0),
        lightness=spec.palette_tweaks.get("lightness", 0),
    )

    supercoords = [v * spec.supersample for v in spec.coords]
    gparams = compute.grid_params().subtile(*supercoords)
    compute.create_mandel(gparams)
    return compute

@app.post("/tile")
async def tile(req: TileRequest):
    spec = req.spec
    compute = spec_to_compute(spec)
    cachekey = f"""
        {spec.center}
        {spec.diam}
        {spec.size}
        {spec.angle}
        {spec.continuous}
        {spec.iter_limit}
        {spec.coords}
        """
    results = await compute_tile(compute, cachekey)
    data_url = "data:image/png;base64," + base64.b64encode(results.pixels).decode("ascii")
    return {
        "url": data_url,
        "seq": req.seq,
        "stats": results.stats,
    }

@app.post("/render")
async def render(spec: ComputeSpec):
    compute = spec_to_compute(spec)
    data = await compute_render(compute)
    return Response(content=data)


def main():
    uvicorn.run("aptus.web.server:app", host="127.0.0.1", port=8042, reload=True)
