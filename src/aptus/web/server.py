import ast
import asyncio
import base64
import functools
import io
import os
import pathlib

import cachetools
import PIL
import pydantic
import uvicorn

from fastapi import FastAPI, Request
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

# Cache of computed counts. One tile is about 830Kb.
cache_size = int(os.getenv("APTUS_CACHE", "500"))
tile_cache = cachetools.LRUCache(cache_size * 1_000_000, getsizeof=lambda nda: nda.nbytes)

@run_in_executor
def compute_tile(compute, cachekey):
    old = tile_cache.get(cachekey)
    if old is None:
        compute.compute_array()
    else:
        compute.set_counts(old)
    pix = compute.color_mandel()
    if old is None:
        tile_cache[cachekey] = compute.counts
    im = PIL.Image.fromarray(pix)
    fout = io.BytesIO()
    compute.write_image(im, fout)
    data_url = "data:image/png;base64," + base64.b64encode(fout.getvalue()).decode("ascii")
    return data_url

class ComputeSpec(pydantic.BaseModel):
    center: tuple[float, float]
    diam: tuple[float, float]
    size: tuple[int, int]
    coords: tuple[int, int, int, int]
    angle: float
    continuous: bool
    iter_limit: int
    palette: list

class TileRequest(pydantic.BaseModel):
    spec: ComputeSpec
    seq: int

@app.post("/tile")
async def tile(
    req: TileRequest
):
    spec = req.spec
    compute = AptusCompute()
    compute.center = spec.center
    compute.diam = spec.diam
    compute.size = spec.size
    compute.angle = spec.angle
    compute.continuous = spec.continuous
    compute.iter_limit = spec.iter_limit
    compute.palette = Palette().from_spec(spec.palette)

    gparams = compute.grid_params().subtile(*spec.coords)
    compute.create_mandel(gparams)

    cachekey = f"""
        {spec.center}
        {spec.diam}
        {spec.size}
        {spec.angle}
        {spec.continuous}
        {spec.iter_limit}
        {spec.coords}
        """
    data_url = await compute_tile(compute, cachekey)
    return {
        "url": data_url,
        "seq": req.seq,
    }

def main():
    uvicorn.run(app, host="127.0.0.1", port=8042)
