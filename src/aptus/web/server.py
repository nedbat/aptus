import ast
import asyncio
import base64
import functools
import io
import os
import pathlib

import PIL
import pydantic
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from aptus.compute import AptusCompute
from aptus.palettes import all_palettes

app = FastAPI()

HERE = pathlib.Path(__file__).parent
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open(HERE / "static/mainpage.html") as f:
        return f.read()

def run_in_executor(f):
    # from https://stackoverflow.com/a/53719009/14343
    @functools.wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, lambda: f(*args, **kwargs))
    return inner

@run_in_executor
def compute_tile(compute):
    compute.compute_array()
    pix = compute.color_mandel()
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
    continuous: bool
    iter_limit: int

@app.post("/tile")
async def tile(
    spec: ComputeSpec
):
    compute = AptusCompute()
    compute.center = spec.center
    compute.diam = spec.diam
    compute.size = spec.size
    compute.continuous = spec.continuous
    compute.iter_limit = spec.iter_limit

    gparams = compute.grid_params().subtile(*spec.coords)
    compute.create_mandel(gparams)

    data_url = await compute_tile(compute)
    return {"url": data_url}

@app.get("/palettes")
async def palettes():
    return [p.spec() for p in all_palettes]

def main():
    uvicorn.run(app, host="127.0.0.1", port=8042)
